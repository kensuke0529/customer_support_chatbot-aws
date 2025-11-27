# +++++++++++++++++++++++++++++
# Imports and Setup
# +++++++++++++++++++++++++++++

import os
import sys
from pathlib import Path
import numpy as np
import boto3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from datetime import datetime

# Add project root to path so we can import from db module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from state import ChatbotInfo
from prompts import CLASSIFICATION_PROMPT, get_response_prompt
from supabase_client import (
    add_data_to_supabase,
    add_embeddings_to_supabase,
    search_similar_documents,
    check_embeddings_exist,
    clear_document_embeddings,
)
from db.escalations_rds import create_escalation, update_escalation_with_contact_info
from db.chat_memory_dynamo import append_message, get_history

# Load environment variables
parent_env = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(parent_env)
load_dotenv()  # Also try current directory
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. "
        "Please set it in your .env file or as an environment variable."
    )

# +++++++++++++++++++++++++++++
# Classification Node
# +++++++++++++++++++++++++++++

classification_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
    temperature=0,
)

chain = CLASSIFICATION_PROMPT | classification_llm


def classify_intent(state: ChatbotInfo):
    """Classifies the user's intent from their message."""
    response = chain.invoke({"input": state.user_message})
    response = response.content
    result = json.loads(response)

    return {
        "classification_tag": result.get("intent", ""),
    }


# +++++++++++++++++++++++++++++
# User Information Extraction Node
# +++++++++++++++++++++++++++++


def extract_user_info(state: ChatbotInfo):
    """
    Extracts user information (email, name, order_id) from conversation messages.
    Uses LLM to intelligently extract structured data from natural language.
    Also stores user message in DynamoDB for chat history.
    """
    # Store user message in DynamoDB
    if state.session_id:
        try:
            append_message(state.session_id, "user", state.user_message)
        except Exception as e:
            print(f"⚠️  Error storing user message in DynamoDB: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("⚠️  session_id is None, skipping DynamoDB storage for user message")

    extraction_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0,
    )

    # Build conversation context for extraction
    conversation_text = ""
    if state.messages:
        # Use all messages for better extraction context
        for msg in state.messages:
            if hasattr(msg, "content"):
                if msg.__class__.__name__ == "HumanMessage":
                    conversation_text += f"User: {msg.content}\n"
                elif msg.__class__.__name__ == "AIMessage":
                    conversation_text += f"Assistant: {msg.content}\n"

    # Also include current message
    conversation_text += f"User: {state.user_message}\n"

    extraction_prompt = f"""Extract structured information from this customer support conversation.

CONVERSATION:
{conversation_text}

Extract the following information if mentioned:
- Email address (any email mentioned)
- Customer name (first name, last name, or full name)
- Order ID or reference number (order numbers, transaction IDs, invoice numbers, etc.)
- Account identifier (account number, username, etc.)

Return ONLY valid JSON (no markdown code blocks):
{{
    "email": "email@example.com" or null,
    "name": "John Doe" or null,
    "order_id": "ORD-12345" or null,
    "account_id": "account123" or null
}}

If information is not found, use null. Be precise - only extract if explicitly mentioned.
"""

    try:
        try:
            response = extraction_llm.invoke(extraction_prompt)
            content = response.content.strip()
        except Exception as e:
            print(f"⚠️  Error calling extraction LLM: {e}")
            import traceback

            traceback.print_exc()
            return {}  # Return empty updates if LLM call fails

        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            print(
                f"⚠️  JSON parsing error in extraction (content: {content[:200]}): {e}"
            )
            # Try to extract email manually as fallback
            import re

            email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            emails = re.findall(email_pattern, state.user_message)
            if emails:
                return {"user_email": emails[0], "contact_info_source": "extracted"}
            return {}

        # Update only if new information is found (don't overwrite existing)
        updates = {}
        contact_found = False

        if result.get("email") and not state.user_email:
            updates["user_email"] = result["email"]
            contact_found = True
        if result.get("name") and not state.user_name:
            updates["user_name"] = result["name"]
        if result.get("order_id") and not state.order_id:
            updates["order_id"] = result["order_id"]

        # Update contact_info_source if we extracted something
        if contact_found and state.contact_info_source == "none":
            updates["contact_info_source"] = "extracted"

        # If we extracted new contact info and have session/thread ID, try to update any existing escalation records in RDS
        if updates and (state.session_id or state.thread_id):
            try:
                update_escalation_with_contact_info(
                    session_id=state.session_id,
                    thread_id=state.thread_id,
                    user_email=updates.get("user_email"),
                    user_name=updates.get("user_name"),
                    order_id=updates.get("order_id"),
                    contact_info_source=updates.get("contact_info_source", "extracted"),
                )
            except Exception as e:
                # Don't fail the entire request if escalation update fails
                print(
                    f"⚠️  Error updating escalation with contact info (non-fatal): {e}"
                )
                import traceback

                traceback.print_exc()

        return updates

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in extraction: {e}")
        return {}
    except Exception as e:
        print(f"Extraction error: {e}")
        return {}


# +++++++++++++++++++++++++++++
# Context Loader (Utility)
# +++++++++++++++++++++++++++++


def doc_loader(
    file_path: str, clear_existing: bool = False, return_chunks: bool = False
):
    """
    Loads PDF documents and stores embeddings in Supabase vector database.

    Args:
        file_path: Path to the PDF file
        clear_existing: If True, clears existing embeddings for this document before adding new ones
        return_chunks: If True, returns list of text chunks instead of count

    Returns:
        Number of chunks processed (if return_chunks=False) or list of text chunks (if return_chunks=True)
    """
    loader = PyPDFLoader(file_path, mode="single")
    docs = loader.load()

    doc_length = len(docs[0].page_content)
    print(f"Document length: {doc_length} characters")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Increased for better context retention
        chunk_overlap=200,  # Increased overlap for better continuity
        length_function=len,
    )

    texts = text_splitter.create_documents([docs[0].page_content])
    print(f"Created {len(texts)} chunks from {file_path}")

    # Get document name from file path
    document_name = os.path.basename(file_path)

    # Check if embeddings already exist for this document
    if clear_existing or not check_embeddings_exist(document_name):
        if clear_existing:
            # Clear existing embeddings for this document
            print(f"Clearing existing embeddings for {document_name}...")
            clear_document_embeddings(document_name)

        # Generate embeddings
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=openai_api_key
        )

        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings_list = embeddings_model.embed_documents(
            [text.page_content for text in texts]
        )

        # Prepare data for Supabase
        embeddings_data = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings_list)):
            embeddings_data.append(
                {
                    "content": text.page_content,
                    "embedding": embedding,
                    "document_name": document_name,
                    "chunk_index": i,
                    "metadata": {},
                }
            )

        # Store in Supabase
        if clear_existing:
            # Delete existing embeddings for this document first
            # We'll need to handle this via a direct query or RPC function
            # For now, we'll insert and handle duplicates at the database level
            pass

        add_embeddings_to_supabase(embeddings_data)
        print(f"✅ Stored {len(texts)} chunks in Supabase for {document_name}")
    else:
        print(f"Embeddings already exist for {document_name}, skipping...")

    # Return chunks if requested, otherwise return count
    if return_chunks:
        return [text.page_content for text in texts]
    return len(texts)


# +++++++++++++++++++++++++++++
# Context Retrieval Node
# +++++++++++++++++++++++++++++

# S3 Vectors config (matching load_documents.py)
S3_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_VECTOR_BUCKET = os.getenv("VECTOR_BUCKET", "s3-vector-chatbot-policy-docs")
S3_VECTOR_INDEX = os.getenv("VECTOR_INDEX", "my-s3-vector-index")

# Initialize S3 Vectors client
s3v_client = boto3.client("s3vectors", region_name=S3_REGION)


def retrieve_context(state: ChatbotInfo):
    """
    Retrieves relevant context from S3 Vectors based on user query.
    Returns empty context if S3 Vectors is not configured or fails.
    """
    try:
        # Try S3 Vectors first
        return retrieve_context_rag(state)
    except Exception as e:
        print(f"S3 Vectors retrieval failed: {e}, returning empty context")
        return {"context": ""}


def retrieve_context_rag(state: ChatbotInfo, top_k: int = 5):
    """
    Retrieves relevant context from S3 Vectors vector database based on user query.

    Args:
        state: ChatbotInfo state containing user message and classification
        top_k: Number of top similar chunks to retrieve

    Returns:
        Dictionary with "context" key containing retrieved text chunks
    """
    # Create embeddings model
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=openai_api_key
    )

    # Build enhanced query with classification context
    query_text = f"{state.classification_tag} question: {state.user_message}"

    # 1. Embed user query
    query_embedding = embeddings_model.embed_query(query_text)
    # Convert to float32 list for S3 Vectors
    query_vec = np.array(query_embedding, dtype=np.float32).tolist()

    # 2. Query S3 Vectors
    try:
        resp = s3v_client.query_vectors(
            vectorBucketName=S3_VECTOR_BUCKET,
            indexName=S3_VECTOR_INDEX,
            queryVector={"float32": query_vec},
            topK=top_k,
            returnMetadata=True,  # Get associated metadata (doc name, chunk text, etc)
        )

        hits = resp.get("vectors", [])
        if not hits:
            print("⚠️  No similar vectors found via S3 Vectors")
            return {"context": ""}

        # 3. Extract text from metadata for each hit
        context_chunks = []
        for hit in hits:
            meta = hit.get("metadata", {})
            # Text is stored in metadata when we store vectors
            text = meta.get("text")
            if text:
                context_chunks.append(text)
            else:
                # Fallback: try to reconstruct from source_doc and chunk_index if text not in metadata
                doc_name = meta.get("source_doc")
                chunk_idx = meta.get("chunk_index")
                if doc_name and chunk_idx is not None:
                    print(
                        f"⚠️  Text not found in metadata for {doc_name} chunk {chunk_idx}"
                    )

        if context_chunks:
            context = "\n\n".join(context_chunks)
            print(f"✅ Retrieved {len(context_chunks)} chunks from S3 Vectors")
            return {"context": context}
        else:
            print("⚠️  No text content found in retrieved vectors")
            return {"context": ""}

    except Exception as e:
        print(f"Error querying S3 Vectors: {e}")
        import traceback

        traceback.print_exc()
        return {"context": ""}


# +++++++++++++++++++++++++++++
# Response Generation Node
# +++++++++++++++++++++++++++++


def generate_response(state: ChatbotInfo):
    """Generates a response based on retrieved context, user query, and conversation history."""
    response_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0.0,
        max_tokens=300,
    )

    # Load recent history from DynamoDB if session_id is available
    conversation_context = ""
    history_loaded = False

    if state.session_id:
        try:
            # Get last 20 messages from DynamoDB (more comprehensive history)
            past_msgs = get_history(state.session_id, limit=20)

            if past_msgs and len(past_msgs) > 0:
                conversation_context = "\n\nPrevious conversation:\n"
                for m in past_msgs:
                    if m["role"] == "user":
                        conversation_context += f"User: {m['content']}\n"
                    elif m["role"] in ("assistant", "human"):
                        conversation_context += f"Assistant: {m['content']}\n"
                history_loaded = True
                print(f"✅ Loaded {len(past_msgs)} messages from DynamoDB")
        except Exception as e:
            print(f"⚠️  Error loading history from DynamoDB: {e}")
            import traceback

            traceback.print_exc()

    # Fallback to state.messages if DynamoDB didn't work or returned empty
    if not history_loaded and state.messages and len(state.messages) > 1:
        recent_messages = state.messages[-10:]  # Get more messages from state
        conversation_context = "\n\nPrevious conversation:\n"
        for msg in recent_messages:
            if hasattr(msg, "content"):
                if msg.__class__.__name__ == "HumanMessage":
                    role = "User"
                elif msg.__class__.__name__ == "AIMessage":
                    role = "Assistant"
                else:
                    role = "System"
                conversation_context += f"{role}: {msg.content}\n"
        print(f"✅ Using {len(recent_messages)} messages from state.messages")

    # Check if user is asking about escalation and add escalation context
    user_message_lower = state.user_message.lower()
    if any(
        keyword in user_message_lower
        for keyword in [
            "escalation",
            "escalate",
            "escalated",
            "support team",
            "human agent",
            "escalation reason",
        ]
    ):
        # Try to get escalation information from RDS
        if state.session_id or state.thread_id:
            try:
                from db.escalations_rds import get_conn, get_table_columns

                columns = get_table_columns("escalations")
                if columns:
                    column_names = {col[0] for col in columns}
                    conn = get_conn()
                    try:
                        with conn.cursor() as cur:
                            # Build query based on available columns
                            select_cols = []
                            if "user_message" in column_names:
                                select_cols.append("user_message")
                            if "classification_tag" in column_names:
                                select_cols.append("classification_tag")
                            if "issue_type" in column_names:
                                select_cols.append("issue_type")

                            if select_cols:
                                order_by_col = None
                                for col in ["created_at", "updated_at", "timestamp"]:
                                    if col in column_names:
                                        order_by_col = col
                                        break

                                where_clause = ""
                                params = []
                                if state.session_id and "session_id" in column_names:
                                    where_clause = "WHERE session_id = %s"
                                    params.append(state.session_id)
                                elif state.thread_id and "thread_id" in column_names:
                                    where_clause = "WHERE thread_id = %s"
                                    params.append(state.thread_id)

                                order_by = (
                                    f"ORDER BY {order_by_col} DESC"
                                    if order_by_col
                                    else ""
                                )
                                sql = f"SELECT {', '.join(select_cols)} FROM escalations {where_clause} {order_by} LIMIT 1"

                                cur.execute(sql, tuple(params) if params else None)
                                escalation = cur.fetchone()

                                if escalation:
                                    escalation_reason = escalation.get(
                                        "user_message", escalation.get("issue_type", "")
                                    )
                                    classification = escalation.get(
                                        "classification_tag",
                                        escalation.get("issue_type", ""),
                                    )
                                    if escalation_reason:
                                        conversation_context += (
                                            f"\n\nNote: A previous query was escalated to human support. The escalated query was: '{escalation_reason}'"
                                            + (
                                                f" (classified as: {classification})"
                                                if classification
                                                else ""
                                            )
                                            + "."
                                        )
                    finally:
                        conn.close()
            except Exception as e:
                print(f"⚠️  Error querying escalation info: {e}")
                import traceback

                traceback.print_exc()
                # Don't fail if we can't get escalation info

    # Check if we have contact info
    has_contact_info = bool(state.user_email or state.user_name)

    # Determine if we should ask for contact info
    # Ask if: no contact info AND (complex issue OR billing/subscription issue OR user seems to need follow-up)
    should_ask_for_info = (
        not has_contact_info
        and state.classification_tag
        in ["billing", "subscription", "account", "returns"]
        and len(state.messages) >= 2  # After at least one exchange
    )

    prompt = get_response_prompt(
        state.context,
        state.user_message,
        conversation_history=conversation_context,
        has_contact_info=has_contact_info,
        should_ask_for_info=should_ask_for_info,
    )
    response = response_llm.invoke(prompt)

    # Post-process: Remove any escalation language that might have slipped through
    response_text = response.content

    # Remove common escalation phrases
    escalation_phrases = [
        "I'll escalate this to our support team for review.",
        "I don't see that in our current policy. I'll escalate this to our support team for review.",
        "I'll escalate this to our support team.",
        "I'll escalate this to the support team.",
        "I'll escalate this to a human agent.",
        "I'll escalate this to our support team for review",
        "I'll escalate this to our support team",
        "I'll escalate this to the support team",
        "I'll escalate this to a human agent",
        "I'll escalate this",
        "I'll escalate",
        "escalate this to our support team",
        "escalate this to the support team",
        "escalate to our support team",
        "escalate to the support team",
        "escalate to a human agent",
    ]

    for phrase in escalation_phrases:
        # Remove the phrase and any trailing punctuation/whitespace
        response_text = response_text.replace(phrase, "").strip()
        # Also handle case variations
        response_text = response_text.replace(phrase.lower(), "").strip()
        response_text = response_text.replace(phrase.capitalize(), "").strip()

    # Clean up any double spaces or periods that might result
    import re

    response_text = re.sub(r"\.\s*\.", ".", response_text)  # Remove double periods
    response_text = re.sub(r"\s+", " ", response_text)  # Remove extra spaces
    response_text = response_text.strip()

    # If response is empty after cleaning, use original (shouldn't happen, but safety net)
    if not response_text:
        response_text = response.content

    # Store assistant response in DynamoDB
    if state.session_id:
        try:
            append_message(state.session_id, "assistant", response_text)
        except Exception as e:
            print(f"⚠️  Error storing assistant message in DynamoDB: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("⚠️  session_id is None, skipping DynamoDB storage for assistant message")

    return {"response": response_text}


# +++++++++++++++++++++++++++++
# Utility: Log Escalation
# +++++++++++++++++++++++++++++


def log_escalation(state: ChatbotInfo, reason: str):
    """Logs escalation data to RDS"""
    # Build metadata dict from validation information
    metadata_dict = {
        "validation_reason": reason,
        "retry_count": state.response_retry_count,
        "context_preview": state.context[:300] if state.context else "",
        "validation_status": state.response_validation,
    }

    try:
        record = create_escalation(
            user_message=state.user_message,
            classification_tag=state.classification_tag,
            response=state.response,
            user_email=state.user_email,
            user_name=state.user_name,
            order_id=state.order_id,
            session_id=state.session_id,
            thread_id=state.thread_id,
            contact_info_source=state.contact_info_source,
            metadata=metadata_dict,
        )

        escalation_id = None
        if record:
            raw_id = record.get("escalation_id") or record.get("id")
            if raw_id:
                escalation_id = str(raw_id)  # Convert to string for consistency
                print(f"✅ Escalation logged to RDS with ID: {escalation_id}")
    except Exception as e:
        print(f"⚠️  Error logging escalation to RDS: {e}")


# +++++++++++++++++++++++++++++
# Response Validation Node
# +++++++++++++++++++++++++++++


def response_validation(state: ChatbotInfo):
    """
    Validates the generated response for quality, accuracy, and completeness.
    Returns PASS, RETRY, or FAIL status.
    """
    validation_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0,
    )

    validation_prompt = f"""You are validating a customer support response. Be lenient and practical.

User Question: {state.user_message}
Generated Response: {state.response}
Context Available: {state.context[:300]}...

Respond in JSON format (NO markdown code blocks, just raw JSON):
{{
    "status": "PASS" | "RETRY" | "FAIL",
    "reason": "Brief explanation"
}}

CRITICAL: CHECK FOR SECURITY EMERGENCIES FIRST!

FAIL IMMEDIATELY if the user question mentions ANY of these:
- Account hacked / hacked account / account is hacked
- Fraud / fraudulent charges / unauthorized charges
- Credit card used without permission
- Unauthorized access / security breach
- Identity theft / stolen account
- Someone else using my account

If user mentions security emergency → ALWAYS return FAIL regardless of response quality.

OTHER RULES:

PASS if:
- Response attempts to answer the question (even if it mentions escalation or says policy info is limited)
- Response is professional enough
- No security emergency mentioned
- Response provides helpful information or guidance, even if partial

RETRY if:
- Response is not friendly or helpful 
- Response is offensive to anyone
- Response is completely unhelpful or nonsensical

FAIL if:
- Security emergency detected (see above)
- Response would harm the customer
- Response is completely empty or contains only errors

IMPORTANT: If a response mentions "escalate" or "I'll escalate", this is still a valid response attempt and should PASS validation. The system will handle actual escalation separately. Only fail if the response is truly unhelpful, offensive, or harmful.

Default to PASS for normal questions. But ALWAYS FAIL for security emergencies.
"""

    try:
        response = validation_llm.invoke(validation_prompt)
        content = response.content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            # Remove opening ```json or ``` and closing ```
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        # Parse JSON
        result = json.loads(content)

        status = result.get("status", "FAIL")
        reason = result.get("reason", "Unknown validation error")

        new_retry_count = state.response_retry_count
        if status == "RETRY":
            new_retry_count += 1

        # Log escalation if security emergency detected
        if status == "FAIL":
            # Check if it's a security emergency based on reason or user message
            security_keywords = [
                "hacked",
                "fraud",
                "unauthorized",
                "security",
                "breach",
                "stolen",
            ]
            user_message_lower = state.user_message.lower()
            reason_lower = reason.lower()

            is_security_emergency = any(
                keyword in user_message_lower or keyword in reason_lower
                for keyword in security_keywords
            )

            if is_security_emergency:
                log_escalation(state, reason)

        return {
            "response_validation": status,
            "response_validation_reason": reason,
            "response_retry_count": new_retry_count,
        }

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in validation: {e}")
        print(f"Raw response content: {response.content[:200]}")
        return {
            "response_validation": "FAIL",
            "response_validation_reason": f"Invalid JSON response from validation: {str(e)}",
            "response_retry_count": state.response_retry_count,
        }
    except Exception as e:
        print(f"Validation error: {e}")
        return {
            "response_validation": "FAIL",
            "response_validation_reason": f"Validation process error: {str(e)}",
            "response_retry_count": state.response_retry_count,
        }


# +++++++++++++++++++++++++++++
# Update Messages Node (for checkpointing)
# +++++++++++++++++++++++++++++


def update_messages_node(state: ChatbotInfo):
    """
    Updates the messages list with the assistant response for checkpointing.
    This ensures conversation history is properly saved.
    """
    from langchain_core.messages import AIMessage

    messages = state.messages.copy() if state.messages else []

    # Add assistant response if it exists and hasn't been added yet
    if state.response and messages:
        # Check if the last message is already the assistant response
        last_msg = messages[-1] if messages else None
        if not (
            last_msg
            and isinstance(last_msg, AIMessage)
            and last_msg.content == state.response
        ):
            messages.append(AIMessage(content=state.response))

    return {"messages": messages}


# +++++++++++++++++++++++++++++
# Escalation Node
# +++++++++++++++++++++++++++++


def escalation_node(state: ChatbotInfo):
    """
    Handles escalation to human support.
    Creates escalation record in RDS and prepares a comprehensive summary for the human agent.
    For security emergencies, provides immediate actionable steps.
    If no contact info found, prompts user for email.
    """
    # Create escalation record in RDS
    # Build metadata dict from additional state information
    metadata_dict = {
        "validation_status": state.response_validation,
        "validation_reason": state.response_validation_reason,
        "retry_count": state.response_retry_count,
        "context_preview": state.context[:300] if state.context else "",
    }

    try:
        record = create_escalation(
            user_message=state.user_message,
            classification_tag=state.classification_tag,
            response=state.response,
            user_email=state.user_email,
            user_name=state.user_name,
            order_id=state.order_id,
            session_id=state.session_id,
            thread_id=state.thread_id,
            contact_info_source=state.contact_info_source,
            metadata=metadata_dict,
        )

        # Extract escalation_id from the returned record
        escalation_id = None
        if record:
            # The record is a RealDictRow, so we can access it like a dict
            raw_id = record.get("escalation_id") or record.get("id")
            if raw_id:
                escalation_id = str(raw_id)  # Convert to string for state compatibility
                print(f"✅ Escalation created in RDS with ID: {escalation_id}")
    except Exception as e:
        print(f"⚠️  Error creating escalation in RDS: {e}")
        escalation_id = None

    # Check if this is a security emergency
    security_keywords = [
        "hacked",
        "hack",
        "fraud",
        "fraudulent",
        "unauthorized",
        "security",
        "breach",
        "stolen",
        "identity theft",
    ]
    user_message_lower = state.user_message.lower()
    is_security_emergency = any(
        keyword in user_message_lower for keyword in security_keywords
    )

    # Check if we need to ask for contact information
    needs_contact = not state.user_email and not state.user_name

    # Build escalation message with user info
    user_info_section = ""
    if state.user_email or state.user_name or state.order_id:
        user_info_section = "\nUser Information:\n"
        if state.user_email:
            user_info_section += (
                f"- Email: {state.user_email} ({state.contact_info_source})\n"
            )
        if state.user_name:
            user_info_section += (
                f"- Name: {state.user_name} ({state.contact_info_source})\n"
            )
        if state.order_id:
            user_info_section += f"- Order ID: {state.order_id}\n"
        if state.session_id:
            user_info_section += f"- Session ID: {state.session_id}\n"

    escalation_message = f"""
🔴 ESCALATION TO HUMAN SUPPORT

User Query: {state.user_message}
{user_info_section}
Classification:
- Tag: {state.classification_tag}

Attempted Response:
{state.response}

Validation Status: {state.response_validation}
Validation Reason: {state.response_validation_reason}
Retry Attempts: {state.response_retry_count}

Context Used:
{state.context[:300]}...

---
This query has been escalated to a human agent for handling.
Please review the context and provide appropriate assistance to the customer.
"""

    # Response message - provide immediate steps for security, then ask for email if needed
    if is_security_emergency:
        # Security emergencies get immediate actionable steps
        response_message = (
            "⚠️ **Immediate Action Required**\n\n"
            "Please take these steps right away to secure your account:\n\n"
            "1. **Report this immediately** to security@company.com\n"
            "2. **Change your password** immediately: Settings > Account > Password\n"
            "3. **Enable Two-Factor Authentication (2FA)** for added security: Settings > Security > 2FA\n"
            "4. **Review recent account activity** for any unauthorized actions\n\n"
        )
        if needs_contact:
            response_message += (
                "We've escalated this to our security team. To ensure we can follow up with you, "
                "could you please provide your email address? A security specialist will contact you shortly."
            )
        else:
            response_message += (
                "We've escalated this to our security team. A security specialist will review your account "
                "and contact you shortly to provide additional assistance."
            )
        needs_contact_info = needs_contact
    elif needs_contact:
        response_message = (
            "Thank you for your patience. Your query has been escalated to our support team. "
            "To ensure we can follow up with you, could you please provide your email address? "
            "A human agent will assist you shortly."
        )
        needs_contact_info = True
    else:
        response_message = (
            "Thank you for your patience. Your query has been escalated to our support team. "
            "A human agent will assist you shortly."
        )
        needs_contact_info = False

    # Store escalation response in DynamoDB
    if state.session_id:
        try:
            append_message(state.session_id, "assistant", response_message)
        except Exception as e:
            print(f"⚠️  Error storing escalation message in DynamoDB: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("⚠️  session_id is None, skipping DynamoDB storage for escalation message")

    # Update messages list for checkpointing (important for conversation history)
    from langchain_core.messages import AIMessage

    messages = state.messages.copy() if state.messages else []
    # Add escalation response to messages if not already there
    if response_message:
        last_msg = messages[-1] if messages else None
        if not (
            last_msg
            and isinstance(last_msg, AIMessage)
            and last_msg.content == response_message
        ):
            messages.append(AIMessage(content=response_message))

    return {
        "response": response_message,
        "messages": messages,  # Include updated messages for checkpointing
        "escalation_summary": escalation_message,
        "needs_contact_info": needs_contact_info,
        "escalation_id": escalation_id,
    }


# +++++++++++++++++++++++++++++
# Utility: Load Documents
# +++++++++++++++++++++++++++++
# Uncomment and run this to load your documents:
"""
if __name__ == "__main__":
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    docs = [
        os.path.join(project_root, "document", "Account Management Policy.pdf"),
        os.path.join(project_root, "document", "Billing & Payment.pdf"),
        os.path.join(project_root, "document", "Contact Information.pdf"),
        os.path.join(project_root, "document", "Customer Support Policy.pdf"),
        os.path.join(project_root, "document", "Shipping & Delivery Policy.pdf"),
        os.path.join(project_root, "document", "Subscription Management Policy.pdf"),
    ]

    vector_store = None
    for doc in docs:
        vector_store = doc_loader(doc, vector_store)

    print("All documents loaded successfully!")
"""
