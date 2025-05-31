# rag_enhanced_banking_architecture.py
"""
Enhanced Banking Compliance Architecture with RAG-based Agent Summarization
and Agentic Pandas Query Processing

This architecture combines:
1. RAG systems for efficient agent knowledge retrieval and summarization
2. Agentic processing for complex compliance analysis with pandas queries
3. Hybrid memory integration for contextual awareness
4. Event-driven communication between components
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import uuid
import json
from pathlib import Path

# Vector store and embeddings
try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain.agents import AgentType

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Enhanced imports for agentic processing
try:
    from smolagents import CodeAgent, Tool

    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes for the hybrid architecture"""
    RAG_ONLY = "rag_only"
    AGENTIC_ONLY = "agentic_only"
    HYBRID = "hybrid"
    AUTO_SELECT = "auto_select"


class AgentRole(Enum):
    """Agent roles in the system"""
    RAG_SUMMARIZER = "rag_summarizer"
    COMPLIANCE_ANALYZER = "compliance_analyzer"
    DORMANCY_PROCESSOR = "dormancy_processor"
    RISK_ASSESSOR = "risk_assessor"
    PANDAS_QUERY_AGENT = "pandas_query_agent"
    COORDINATOR = "coordinator"


@dataclass
class AgentSummary:
    """Structured agent summary for RAG storage"""
    agent_id: str
    agent_role: AgentRole
    summary_text: str
    key_capabilities: List[str]
    last_execution_results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    knowledge_tags: List[str]
    created_at: datetime
    updated_at: datetime
    execution_count: int = 0
    success_rate: float = 1.0


@dataclass
class QueryContext:
    """Context for query processing"""
    session_id: str
    query_type: str
    data_shape: Tuple[int, int]
    required_capabilities: List[str]
    complexity_score: float
    processing_mode: ProcessingMode
    memory_context: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingResult:
    """Result from processing pipeline"""
    result_id: str
    agent_role: AgentRole
    processing_mode: ProcessingMode
    data: Dict[str, Any]
    confidence_score: float
    processing_time: float
    memory_usage_mb: float
    status: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGAgentKnowledgeBase:
    """RAG-based knowledge base for agent summaries and capabilities"""

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
        self.vector_store: Optional[FAISS] = None
        self.embeddings = None
        self.agent_summaries: Dict[str, AgentSummary] = {}
        self.knowledge_base_path = "data/rag_agent_kb"

        # Initialize components
        if LANGCHAIN_AVAILABLE:
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " "]
            )

        # Create storage directory
        Path(self.knowledge_base_path).mkdir(parents=True, exist_ok=True)

        # Load existing knowledge base
        asyncio.create_task(self._load_knowledge_base())

    async def add_agent_summary(self, summary: AgentSummary) -> str:
        """Add or update agent summary in RAG knowledge base"""
        try:
            # Store summary
            self.agent_summaries[summary.agent_id] = summary

            # Create document for vector storage
            doc_content = self._create_summary_document(summary)
            document = Document(
                page_content=doc_content,
                metadata={
                    "agent_id": summary.agent_id,
                    "agent_role": summary.agent_role.value,
                    "created_at": summary.created_at.isoformat(),
                    "success_rate": summary.success_rate,
                    "execution_count": summary.execution_count
                }
            )

            # Add to vector store
            if self.vector_store is None and LANGCHAIN_AVAILABLE:
                self.vector_store = FAISS.from_documents([document], self.embeddings)
            elif LANGCHAIN_AVAILABLE:
                self.vector_store.add_documents([document])

            # Persist to disk
            await self._save_knowledge_base()

            logger.info(f"Added agent summary for {summary.agent_id} to RAG knowledge base")
            return summary.agent_id

        except Exception as e:
            logger.error(f"Failed to add agent summary: {e}")
            raise

    async def query_agent_capabilities(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query agent capabilities using RAG"""
        try:
            if not self.vector_store or not LANGCHAIN_AVAILABLE:
                return self._fallback_capability_search(query, top_k)

            # Perform semantic search
            docs = self.vector_store.similarity_search_with_score(query, k=top_k)

            results = []
            for doc, score in docs:
                agent_id = doc.metadata.get("agent_id")
                if agent_id in self.agent_summaries:
                    summary = self.agent_summaries[agent_id]
                    results.append({
                        "agent_id": agent_id,
                        "agent_role": summary.agent_role,
                        "summary": summary.summary_text,
                        "capabilities": summary.key_capabilities,
                        "success_rate": summary.success_rate,
                        "relevance_score": 1.0 - score,  # Convert distance to similarity
                        "metadata": doc.metadata
                    })

            return results

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return self._fallback_capability_search(query, top_k)

    async def get_recommended_agents(self, context: QueryContext) -> List[str]:
        """Get recommended agents for a query context"""
        try:
            # Build query from context
            query_parts = [
                f"query type: {context.query_type}",
                f"data shape: {context.data_shape}",
                f"capabilities: {', '.join(context.required_capabilities)}"
            ]
            query = " ".join(query_parts)

            # Query for relevant agents
            capabilities = await self.query_agent_capabilities(query, top_k=5)

            # Score and rank agents
            recommendations = []
            for cap in capabilities:
                score = self._calculate_agent_score(cap, context)
                if score > 0.3:  # Minimum relevance threshold
                    recommendations.append({
                        "agent_id": cap["agent_id"],
                        "score": score,
                        "reasoning": self._generate_recommendation_reasoning(cap, context)
                    })

            # Sort by score
            recommendations.sort(key=lambda x: x["score"], reverse=True)

            return [rec["agent_id"] for rec in recommendations[:3]]

        except Exception as e:
            logger.error(f"Failed to get agent recommendations: {e}")
            return []

    def _create_summary_document(self, summary: AgentSummary) -> str:
        """Create text document from agent summary"""
        return f"""
Agent Role: {summary.agent_role.value}
Summary: {summary.summary_text}
Key Capabilities: {', '.join(summary.key_capabilities)}
Performance Metrics: Success Rate: {summary.success_rate:.2%}, Executions: {summary.execution_count}
Knowledge Tags: {', '.join(summary.knowledge_tags)}
Last Updated: {summary.updated_at.isoformat()}
"""

    def _fallback_capability_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback search when vector store is unavailable"""
        results = []
        query_lower = query.lower()

        for agent_id, summary in self.agent_summaries.items():
            # Simple text matching
            score = 0.0
            text_to_search = (summary.summary_text + " " + " ".join(summary.key_capabilities)).lower()

            query_words = query_lower.split()
            for word in query_words:
                if word in text_to_search:
                    score += 1.0 / len(query_words)

            if score > 0:
                results.append({
                    "agent_id": agent_id,
                    "agent_role": summary.agent_role,
                    "summary": summary.summary_text,
                    "capabilities": summary.key_capabilities,
                    "success_rate": summary.success_rate,
                    "relevance_score": score,
                    "metadata": {"fallback_search": True}
                })

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

    def _calculate_agent_score(self, capability: Dict[str, Any], context: QueryContext) -> float:
        """Calculate agent recommendation score"""
        score = 0.0

        # Base relevance score
        score += capability["relevance_score"] * 0.4

        # Success rate factor
        score += capability["success_rate"] * 0.3

        # Capability matching
        agent_caps = set(cap.lower() for cap in capability["capabilities"])
        required_caps = set(cap.lower() for cap in context.required_capabilities)
        capability_overlap = len(agent_caps.intersection(required_caps)) / max(len(required_caps), 1)
        score += capability_overlap * 0.3

        return min(score, 1.0)

    def _generate_recommendation_reasoning(self, capability: Dict[str, Any], context: QueryContext) -> str:
        """Generate reasoning for agent recommendation"""
        reasons = []

        if capability["relevance_score"] > 0.8:
            reasons.append("High semantic relevance to query")

        if capability["success_rate"] > 0.9:
            reasons.append("Excellent historical performance")

        agent_caps = set(cap.lower() for cap in capability["capabilities"])
        required_caps = set(cap.lower() for cap in context.required_capabilities)
        if agent_caps.intersection(required_caps):
            reasons.append("Matches required capabilities")

        return "; ".join(reasons) if reasons else "General compatibility"

    async def _load_knowledge_base(self):
        """Load knowledge base from disk"""
        try:
            # Load agent summaries
            summaries_file = Path(self.knowledge_base_path) / "agent_summaries.json"
            if summaries_file.exists():
                with open(summaries_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        summary = AgentSummary(
                            agent_id=item["agent_id"],
                            agent_role=AgentRole(item["agent_role"]),
                            summary_text=item["summary_text"],
                            key_capabilities=item["key_capabilities"],
                            last_execution_results=item["last_execution_results"],
                            performance_metrics=item["performance_metrics"],
                            knowledge_tags=item["knowledge_tags"],
                            created_at=datetime.fromisoformat(item["created_at"]),
                            updated_at=datetime.fromisoformat(item["updated_at"]),
                            execution_count=item["execution_count"],
                            success_rate=item["success_rate"]
                        )
                        self.agent_summaries[summary.agent_id] = summary

            # Load vector store
            if LANGCHAIN_AVAILABLE:
                vector_store_path = Path(self.knowledge_base_path) / "vector_store"
                if vector_store_path.exists():
                    self.vector_store = FAISS.load_local(str(vector_store_path), self.embeddings)

            logger.info("RAG knowledge base loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load knowledge base: {e}")

    async def _save_knowledge_base(self):
        """Save knowledge base to disk"""
        try:
            # Save agent summaries
            summaries_data = []
            for summary in self.agent_summaries.values():
                summaries_data.append({
                    "agent_id": summary.agent_id,
                    "agent_role": summary.agent_role.value,
                    "summary_text": summary.summary_text,
                    "key_capabilities": summary.key_capabilities,
                    "last_execution_results": summary.last_execution_results,
                    "performance_metrics": summary.performance_metrics,
                    "knowledge_tags": summary.knowledge_tags,
                    "created_at": summary.created_at.isoformat(),
                    "updated_at": summary.updated_at.isoformat(),
                    "execution_count": summary.execution_count,
                    "success_rate": summary.success_rate
                })

            summaries_file = Path(self.knowledge_base_path) / "agent_summaries.json"
            with open(summaries_file, 'w') as f:
                json.dump(summaries_data, f, indent=2)

            # Save vector store
            if self.vector_store and LANGCHAIN_AVAILABLE:
                vector_store_path = Path(self.knowledge_base_path) / "vector_store"
                self.vector_store.save_local(str(vector_store_path))

        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")


class PandasQueryAgent:
    """Enhanced Pandas query agent with RAG integration"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.agent = None
        self.current_dataframe = None
        self.query_history: List[Dict[str, Any]] = []

        # Performance metrics
        self.execution_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_execution_time": 0.0,
            "common_operations": {}
        }

    async def initialize_with_dataframe(self, df: pd.DataFrame, context: QueryContext) -> bool:
        """Initialize agent with dataframe and context"""
        try:
            if not LANGCHAIN_AVAILABLE:
                logger.warning("LangChain not available, using fallback pandas processing")
                self.current_dataframe = df
                return True

            # Create pandas agent
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=0,
                max_tokens=1000
            )

            self.agent = create_pandas_dataframe_agent(
                llm=llm,
                df=df,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                verbose=True,
                allow_dangerous_code=True,  # Required for pandas operations
                max_iterations=5,
                return_intermediate_steps=True
            )

            self.current_dataframe = df

            logger.info(f"Pandas agent initialized with dataframe shape: {df.shape}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize pandas agent: {e}")
            self.current_dataframe = df  # Fallback to basic pandas
            return False

    async def execute_query(self, query: str, session_id: str) -> ProcessingResult:
        """Execute pandas query with agentic processing"""
        start_time = datetime.now()

        try:
            # Record query
            query_record = {
                "query": query,
                "timestamp": start_time.isoformat(),
                "session_id": session_id
            }

            if self.agent and LANGCHAIN_AVAILABLE:
                # Use LangChain agent
                result = await self._execute_with_langchain_agent(query, session_id)
            else:
                # Fallback to direct pandas processing
                result = await self._execute_with_direct_pandas(query, session_id)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            # Update statistics
            self._update_execution_stats(query, execution_time, result["success"])

            # Create processing result
            processing_result = ProcessingResult(
                result_id=str(uuid.uuid4()),
                agent_role=AgentRole.PANDAS_QUERY_AGENT,
                processing_mode=ProcessingMode.AGENTIC_ONLY,
                data=result,
                confidence_score=result.get("confidence", 0.8),
                processing_time=execution_time,
                memory_usage_mb=self._estimate_memory_usage(),
                status="success" if result["success"] else "error",
                error=result.get("error"),
                metadata={
                    "query": query,
                    "dataframe_shape": self.current_dataframe.shape if self.current_dataframe is not None else None,
                    "agent_type": "langchain" if self.agent else "direct_pandas"
                }
            )

            # Store query record
            query_record.update({
                "result": result,
                "execution_time": execution_time,
                "success": result["success"]
            })
            self.query_history.append(query_record)

            return processing_result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Query execution failed: {e}")

            return ProcessingResult(
                result_id=str(uuid.uuid4()),
                agent_role=AgentRole.PANDAS_QUERY_AGENT,
                processing_mode=ProcessingMode.AGENTIC_ONLY,
                data={"error": str(e), "success": False},
                confidence_score=0.0,
                processing_time=execution_time,
                memory_usage_mb=0.0,
                status="error",
                error=str(e)
            )

    async def _execute_with_langchain_agent(self, query: str, session_id: str) -> Dict[str, Any]:
        """Execute query using LangChain pandas agent"""
        try:
            # Execute the query
            response = self.agent.invoke({"input": query})

            return {
                "success": True,
                "result": response.get("output", ""),
                "intermediate_steps": response.get("intermediate_steps", []),
                "confidence": 0.9,
                "method": "langchain_agent"
            }

        except Exception as e:
            logger.error(f"LangChain agent execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "langchain_agent"
            }

    async def _execute_with_direct_pandas(self, query: str, session_id: str) -> Dict[str, Any]:
        """Fallback execution using direct pandas operations"""
        try:
            if self.current_dataframe is None:
                raise ValueError("No dataframe available")

            # Simple query interpretation and execution
            result = self._interpret_and_execute_pandas_query(query)

            return {
                "success": True,
                "result": result,
                "confidence": 0.7,
                "method": "direct_pandas"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "direct_pandas"
            }

    def _interpret_and_execute_pandas_query(self, query: str) -> str:
        """Simple query interpretation for common operations"""
        df = self.current_dataframe
        query_lower = query.lower()

        # Basic query patterns
        if "shape" in query_lower or "size" in query_lower:
            return f"DataFrame shape: {df.shape}"

        elif "columns" in query_lower:
            return f"Columns: {list(df.columns)}"

        elif "head" in query_lower:
            return df.head().to_string()

        elif "info" in query_lower:
            return df.info()

        elif "describe" in query_lower:
            return df.describe().to_string()

        elif "null" in query_lower or "missing" in query_lower:
            return df.isnull().sum().to_string()

        elif "unique" in query_lower:
            # Find column name in query
            for col in df.columns:
                if col.lower() in query_lower:
                    return f"Unique values in {col}: {df[col].nunique()}"
            return "Please specify a column for unique values"

        elif "count" in query_lower:
            return f"Row count: {len(df)}"

        else:
            return "Query not recognized. Try: shape, columns, head, info, describe, null values, count, or unique values for a specific column."

    def _update_execution_stats(self, query: str, execution_time: float, success: bool):
        """Update execution statistics"""
        self.execution_stats["total_queries"] += 1

        if success:
            self.execution_stats["successful_queries"] += 1

        # Update average execution time
        total = self.execution_stats["total_queries"]
        current_avg = self.execution_stats["average_execution_time"]
        self.execution_stats["average_execution_time"] = (
                                                                 (current_avg * (total - 1)) + execution_time
                                                         ) / total

        # Track common operations
        query_type = self._classify_query_type(query)
        if query_type in self.execution_stats["common_operations"]:
            self.execution_stats["common_operations"][query_type] += 1
        else:
            self.execution_stats["common_operations"][query_type] = 1

    def _classify_query_type(self, query: str) -> str:
        """Classify query type for statistics"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["count", "len", "size"]):
            return "aggregation"
        elif any(word in query_lower for word in ["filter", "where", "select"]):
            return "filtering"
        elif any(word in query_lower for word in ["group", "aggregate"]):
            return "grouping"
        elif any(word in query_lower for word in ["sort", "order"]):
            return "sorting"
        elif any(word in query_lower for word in ["merge", "join"]):
            return "joining"
        else:
            return "other"

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        if self.current_dataframe is not None:
            return self.current_dataframe.memory_usage(deep=True).sum() / 1024 / 1024
        return 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        success_rate = 0.0
        if self.execution_stats["total_queries"] > 0:
            success_rate = self.execution_stats["successful_queries"] / self.execution_stats["total_queries"]

        return {
            "total_queries": self.execution_stats["total_queries"],
            "success_rate": success_rate,
            "average_execution_time": self.execution_stats["average_execution_time"],
            "common_operations": self.execution_stats["common_operations"],
            "recent_queries": self.query_history[-5:] if self.query_history else []
        }


class HybridComplianceOrchestrator:
    """Main orchestrator combining RAG and agentic processing"""

    def __init__(self):
        self.rag_knowledge_base = RAGAgentKnowledgeBase()
        self.pandas_agent = PandasQueryAgent()
        self.processing_history: List[ProcessingResult] = []
        self.session_contexts: Dict[str, QueryContext] = {}

        # Initialize with default agent summaries
        asyncio.create_task(self._initialize_default_agents())

    async def process_compliance_data(
            self,
            data: pd.DataFrame,
            query: str,
            session_id: str,
            processing_mode: ProcessingMode = ProcessingMode.AUTO_SELECT
    ) -> ProcessingResult:
        """Main processing function combining RAG and agentic approaches"""

        try:
            # Create query context
            context = QueryContext(
                session_id=session_id,
                query_type=self._classify_query_type(query),
                data_shape=data.shape,
                required_capabilities=self._extract_required_capabilities(query),
                complexity_score=self._calculate_complexity_score(query, data),
                processing_mode=processing_mode
            )

            # Store context
            self.session_contexts[session_id] = context

            # Auto-select processing mode if needed
            if processing_mode == ProcessingMode.AUTO_SELECT:
                context.processing_mode = await self._auto_select_processing_mode(context)

            # Process based on mode
            if context.processing_mode == ProcessingMode.RAG_ONLY:
                result = await self._process_with_rag_only(data, query, context)
            elif context.processing_mode == ProcessingMode.AGENTIC_ONLY:
                result = await self._process_with_agentic_only(data, query, context)
            else:  # HYBRID
                result = await self._process_with_hybrid_approach(data, query, context)

            # Store result
            self.processing_history.append(result)

            # Update agent summaries based on results
            await self._update_agent_summaries(result)

            return result

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return ProcessingResult(
                result_id=str(uuid.uuid4()),
                agent_role=AgentRole.COORDINATOR,
                processing_mode=processing_mode,
                data={"error": str(e)},
                confidence_score=0.0,
                processing_time=0.0,
                memory_usage_mb=0.0,
                status="error",
                error=str(e)
            )

    async def _process_with_rag_only(self, data: pd.DataFrame, query: str, context: QueryContext) -> ProcessingResult:
        """Process using only RAG-based retrieval of agent summaries"""
        start_time = datetime.now()

        try:
            # Query for relevant agents
            relevant_agents = await self.rag_knowledge_base.query_agent_capabilities(query, top_k=3)

            # Synthesize response from agent summaries
            synthesized_response = self._synthesize_rag_response(relevant_agents, query, data)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessingResult(
                result_id=str(uuid.uuid4()),
                agent_role=AgentRole.RAG_SUMMARIZER,
                processing_mode=ProcessingMode.RAG_ONLY,
                data={
                    "response": synthesized_response,
                    "relevant_agents": relevant_agents,
                    "method": "rag_synthesis"
                },
                confidence_score=0.7,  # Lower confidence for synthesis only
                processing_time=processing_time,
                memory_usage_mb=data.memory_usage(deep=True).sum() / 1024 / 1024,
                status="success"
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessingResult(
                result_id=str(uuid.uuid4()),
                agent_role=AgentRole.RAG_SUMMARIZER,
                processing_mode=ProcessingMode.RAG_ONLY,
                data={"error": str(e)},
                confidence_score=0.0,
                processing_time=processing_time,
                memory_usage_mb=0.0,
                status="error",
                error=str(e)
            )

    async def _process_with_agentic_only(self, data: pd.DataFrame, query: str,
                                         context: QueryContext) -> ProcessingResult:
        """Process using only agentic pandas processing"""
        try:
            # Initialize pandas agent with data
            await self.pandas_agent.initialize_with_dataframe(data, context)

            # Execute query
            result = await self.pandas_agent.execute_query(query, context.session_id)

            return result

        except Exception as e:
            return ProcessingResult(
                result_id=str(uuid.uuid4()),
                agent_role=AgentRole.PANDAS_QUERY_AGENT,
                processing_mode=ProcessingMode.AGENTIC_ONLY,
                data={"error": str(e)},
                confidence_score=0.0,
                processing_time=0.0,
                memory_usage_mb=0.0,
                status="error",
                error=str(e)
            )

    async def _process_with_hybrid_approach(self, data: pd.DataFrame, query: str,
                                            context: QueryContext) -> ProcessingResult:
        """Process using hybrid RAG + agentic approach"""
        start_time = datetime.now()

        try:
            # Step 1: RAG-based agent selection and context retrieval
            relevant_agents = await self.rag_knowledge_base.query_agent_capabilities(query, top_k=5)
            recommended_agents = await self.rag_knowledge_base.get_recommended_agents(context)

            # Step 2: Initialize pandas agent with enhanced context
            await self.pandas_agent.initialize_with_dataframe(data, context)

            # Step 3: Enhanced query with RAG context
            enhanced_query = self._enhance_query_with_rag_context(query, relevant_agents)

            # Step 4: Execute agentic processing
            agentic_result = await self.pandas_agent.execute_query(enhanced_query, context.session_id)

            # Step 5: Validate and enhance results with RAG knowledge
            validated_result = await self._validate_with_rag_knowledge(agentic_result, relevant_agents)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessingResult(
                result_id=str(uuid.uuid4()),
                agent_role=AgentRole.COORDINATOR,
                processing_mode=ProcessingMode.HYBRID,
                data={
                    "agentic_result": agentic_result.data,
                    "rag_context": relevant_agents,
                    "recommended_agents": recommended_agents,
                    "enhanced_query": enhanced_query,
                    "validation": validated_result,
                    "method": "hybrid_rag_agentic"
                },
                confidence_score=min(agentic_result.confidence_score + 0.2, 1.0),  # Boost confidence with RAG
                processing_time=processing_time,
                memory_usage_mb=agentic_result.memory_usage_mb,
                status="success" if agentic_result.status == "success" else "partial_success",
                metadata={
                    "rag_agents_used": len(relevant_agents),
                    "agentic_confidence": agentic_result.confidence_score,
                    "hybrid_enhancement": True
                }
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Hybrid processing failed: {e}")

            return ProcessingResult(
                result_id=str(uuid.uuid4()),
                agent_role=AgentRole.COORDINATOR,
                processing_mode=ProcessingMode.HYBRID,
                data={"error": str(e)},
                confidence_score=0.0,
                processing_time=processing_time,
                memory_usage_mb=0.0,
                status="error",
                error=str(e)
            )

    async def _auto_select_processing_mode(self, context: QueryContext) -> ProcessingMode:
        """Automatically select the best processing mode based on context"""
        try:
            # Decision factors
            complexity = context.complexity_score
            data_size = context.data_shape[0] * context.data_shape[1]
            capabilities_count = len(context.required_capabilities)

            # Simple decision tree
            if complexity < 0.3 and data_size < 1000:
                # Simple queries on small data - use RAG only
                return ProcessingMode.RAG_ONLY
            elif complexity > 0.8 or data_size > 100000:
                # Complex queries or large data - use hybrid approach
                return ProcessingMode.HYBRID
            elif capabilities_count > 3:
                # Multiple capabilities needed - use hybrid
                return ProcessingMode.HYBRID
            else:
                # Default to agentic for medium complexity
                return ProcessingMode.AGENTIC_ONLY

        except Exception as e:
            logger.warning(f"Auto-selection failed, defaulting to hybrid: {e}")
            return ProcessingMode.HYBRID

    def _synthesize_rag_response(self, relevant_agents: List[Dict[str, Any]], query: str, data: pd.DataFrame) -> str:
        """Synthesize response from RAG agent summaries"""
        if not relevant_agents:
            return "No relevant agent knowledge found for this query."

        # Combine agent summaries
        response_parts = []
        response_parts.append(f"Based on agent knowledge for query: '{query}'")
        response_parts.append(f"Data shape: {data.shape}")
        response_parts.append("")

        for i, agent in enumerate(relevant_agents, 1):
            response_parts.append(f"{i}. Agent: {agent['agent_role'].value}")
            response_parts.append(f"   Relevance: {agent['relevance_score']:.2%}")
            response_parts.append(f"   Summary: {agent['summary']}")
            response_parts.append(f"   Capabilities: {', '.join(agent['capabilities'])}")
            response_parts.append("")

        # Add basic data insights
        response_parts.append("Basic Data Insights:")
        response_parts.append(f"- Total records: {len(data)}")
        response_parts.append(f"- Columns: {len(data.columns)}")
        if 'Account_ID' in data.columns:
            response_parts.append(f"- Unique accounts: {data['Account_ID'].nunique()}")

        return "\n".join(response_parts)

    def _enhance_query_with_rag_context(self, query: str, relevant_agents: List[Dict[str, Any]]) -> str:
        """Enhance query with context from RAG agents"""
        if not relevant_agents:
            return query

        # Extract relevant capabilities and context
        all_capabilities = []
        for agent in relevant_agents:
            all_capabilities.extend(agent['capabilities'])

        # Build enhanced query
        enhanced_parts = [query]

        if all_capabilities:
            enhanced_parts.append(f"Context: Focus on {', '.join(set(all_capabilities))}")

        return ". ".join(enhanced_parts)

    async def _validate_with_rag_knowledge(self, agentic_result: ProcessingResult,
                                           relevant_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate agentic results against RAG knowledge"""
        validation = {
            "validated": True,
            "confidence_boost": 0.0,
            "consistency_checks": [],
            "recommendations": []
        }

        try:
            if agentic_result.status != "success":
                validation["validated"] = False
                validation["recommendations"].append("Agentic processing failed - rely on RAG knowledge")
                return validation

            # Check consistency with agent capabilities
            for agent in relevant_agents:
                if agent['success_rate'] > 0.9:
                    validation["confidence_boost"] += 0.1
                    validation["consistency_checks"].append(
                        f"High-performing agent {agent['agent_id']} supports this analysis")

            # Normalize confidence boost
            validation["confidence_boost"] = min(validation["confidence_boost"], 0.3)

            return validation

        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            validation["validated"] = False
            return validation

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["dormant", "dormancy", "inactive"]):
            return "dormancy_analysis"
        elif any(word in query_lower for word in ["compliance", "regulation", "cbuae"]):
            return "compliance_check"
        elif any(word in query_lower for word in ["risk", "assessment", "score"]):
            return "risk_analysis"
        elif any(word in query_lower for word in ["count", "sum", "total", "aggregate"]):
            return "aggregation"
        elif any(word in query_lower for word in ["filter", "where", "find"]):
            return "filtering"
        else:
            return "general_analysis"

    def _extract_required_capabilities(self, query: str) -> List[str]:
        """Extract required capabilities from query"""
        capabilities = []
        query_lower = query.lower()

        capability_keywords = {
            "dormancy_analysis": ["dormant", "dormancy", "inactive", "activity"],
            "compliance_checking": ["compliance", "regulation", "rules", "cbuae"],
            "data_aggregation": ["count", "sum", "total", "average", "aggregate"],
            "data_filtering": ["filter", "where", "find", "search", "select"],
            "risk_assessment": ["risk", "assessment", "score", "evaluation"],
            "pandas_operations": ["dataframe", "column", "row", "merge", "join"],
            "statistical_analysis": ["mean", "median", "std", "correlation", "distribution"]
        }

        for capability, keywords in capability_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                capabilities.append(capability)

        return capabilities if capabilities else ["general_analysis"]

    def _calculate_complexity_score(self, query: str, data: pd.DataFrame) -> float:
        """Calculate query complexity score (0.0 to 1.0)"""
        score = 0.0
        query_lower = query.lower()

        # Query complexity factors
        complexity_indicators = {
            "multiple operations": ["and", "or", "then", "also"],
            "aggregation": ["group by", "aggregate", "sum", "count", "average"],
            "joining": ["join", "merge", "combine"],
            "statistical": ["correlation", "distribution", "variance", "regression"],
            "time series": ["trend", "time", "period", "temporal"],
            "complex conditions": ["if", "when", "case", "conditional"]
        }

        for category, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                score += 0.15

        # Data size factor
        data_size = data.shape[0] * data.shape[1]
        if data_size > 100000:
            score += 0.2
        elif data_size > 10000:
            score += 0.1

        # Query length factor
        word_count = len(query.split())
        if word_count > 20:
            score += 0.2
        elif word_count > 10:
            score += 0.1

        return min(score, 1.0)

    async def _update_agent_summaries(self, result: ProcessingResult):
        """Update agent summaries based on execution results"""
        try:
            # Create or update agent summary
            agent_id = f"{result.agent_role.value}_{result.result_id[:8]}"

            # Check if agent summary exists
            existing_summary = None
            for summary in self.rag_knowledge_base.agent_summaries.values():
                if summary.agent_role == result.agent_role:
                    existing_summary = summary
                    break

            if existing_summary:
                # Update existing summary
                existing_summary.execution_count += 1
                existing_summary.last_execution_results = result.data
                existing_summary.updated_at = datetime.now()

                # Update success rate
                if result.status == "success":
                    new_successes = existing_summary.success_rate * (existing_summary.execution_count - 1) + 1
                    existing_summary.success_rate = new_successes / existing_summary.execution_count
                else:
                    new_successes = existing_summary.success_rate * (existing_summary.execution_count - 1)
                    existing_summary.success_rate = new_successes / existing_summary.execution_count

                # Update performance metrics
                existing_summary.performance_metrics.update({
                    "avg_processing_time": result.processing_time,
                    "confidence_score": result.confidence_score,
                    "memory_usage_mb": result.memory_usage_mb
                })

                await self.rag_knowledge_base.add_agent_summary(existing_summary)

        except Exception as e:
            logger.warning(f"Failed to update agent summaries: {e}")

    async def _initialize_default_agents(self):
        """Initialize default agent summaries"""
        try:
            default_agents = [
                AgentSummary(
                    agent_id="dormancy_analyzer_default",
                    agent_role=AgentRole.DORMANCY_PROCESSOR,
                    summary_text="Specialized in analyzing account dormancy patterns, identifying inactive accounts, and applying CBUAE dormancy regulations.",
                    key_capabilities=["dormancy_analysis", "inactivity_detection", "regulatory_compliance",
                                      "pattern_recognition"],
                    last_execution_results={},
                    performance_metrics={"avg_processing_time": 0.5, "confidence_score": 0.9},
                    knowledge_tags=["dormancy", "cbuae", "regulations", "banking"],
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                ),
                AgentSummary(
                    agent_id="compliance_checker_default",
                    agent_role=AgentRole.COMPLIANCE_ANALYZER,
                    summary_text="Expert in banking compliance verification, regulatory rule checking, and CBUAE compliance validation.",
                    key_capabilities=["compliance_checking", "regulatory_validation", "rule_enforcement",
                                      "audit_support"],
                    last_execution_results={},
                    performance_metrics={"avg_processing_time": 0.7, "confidence_score": 0.95},
                    knowledge_tags=["compliance", "regulations", "audit", "banking"],
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                ),
                AgentSummary(
                    agent_id="pandas_query_default",
                    agent_role=AgentRole.PANDAS_QUERY_AGENT,
                    summary_text="Advanced pandas data processing agent capable of complex data analysis, aggregations, and statistical operations.",
                    key_capabilities=["pandas_operations", "data_aggregation", "statistical_analysis",
                                      "data_filtering"],
                    last_execution_results={},
                    performance_metrics={"avg_processing_time": 0.3, "confidence_score": 0.85},
                    knowledge_tags=["pandas", "data_analysis", "statistics", "querying"],
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                ),
                AgentSummary(
                    agent_id="risk_assessor_default",
                    agent_role=AgentRole.RISK_ASSESSOR,
                    summary_text="Risk assessment specialist focusing on compliance risk evaluation, confidence scoring, and risk mitigation recommendations.",
                    key_capabilities=["risk_assessment", "confidence_scoring", "risk_mitigation", "evaluation"],
                    last_execution_results={},
                    performance_metrics={"avg_processing_time": 0.4, "confidence_score": 0.88},
                    knowledge_tags=["risk", "assessment", "evaluation", "banking"],
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
            ]

            for agent_summary in default_agents:
                await self.rag_knowledge_base.add_agent_summary(agent_summary)

            logger.info("Default agent summaries initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize default agents: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "rag_knowledge_base": {
                "total_agents": len(self.rag_knowledge_base.agent_summaries),
                "vector_store_available": self.rag_knowledge_base.vector_store is not None,
                "embedding_model": self.rag_knowledge_base.embedding_model
            },
            "pandas_agent": {
                "performance": self.pandas_agent.get_performance_summary(),
                "current_dataframe_shape": self.pandas_agent.current_dataframe.shape if self.pandas_agent.current_dataframe is not None else None
            },
            "processing_history": {
                "total_sessions": len(self.session_contexts),
                "total_results": len(self.processing_history),
                "recent_modes": [r.processing_mode.value for r in self.processing_history[-10:]]
            },
            "capabilities": {
                "langchain_available": LANGCHAIN_AVAILABLE,
                "smolagents_available": SMOLAGENTS_AVAILABLE,
                "supported_modes": [mode.value for mode in ProcessingMode]
            }
        }

    async def get_query_recommendations(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Get recommendations for query processing"""
        try:
            # Create temporary context
            temp_context = QueryContext(
                session_id="temp_recommendation",
                query_type=self._classify_query_type(query),
                data_shape=data.shape,
                required_capabilities=self._extract_required_capabilities(query),
                complexity_score=self._calculate_complexity_score(query, data),
                processing_mode=ProcessingMode.AUTO_SELECT
            )

            # Get recommendations
            recommended_mode = await self._auto_select_processing_mode(temp_context)
            relevant_agents = await self.rag_knowledge_base.query_agent_capabilities(query, top_k=3)

            return {
                "recommended_mode": recommended_mode.value,
                "complexity_score": temp_context.complexity_score,
                "query_type": temp_context.query_type,
                "required_capabilities": temp_context.required_capabilities,
                "relevant_agents": [
                    {
                        "agent_role": agent["agent_role"].value,
                        "relevance": agent["relevance_score"],
                        "success_rate": agent["success_rate"]
                    }
                    for agent in relevant_agents
                ],
                "recommendations": self._generate_processing_recommendations(temp_context, relevant_agents)
            }

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return {"error": str(e)}

    def _generate_processing_recommendations(self, context: QueryContext, relevant_agents: List[Dict[str, Any]]) -> \
    List[str]:
        """Generate processing recommendations"""
        recommendations = []

        # Mode-specific recommendations
        if context.complexity_score < 0.3:
            recommendations.append("Consider RAG-only mode for faster processing of simple queries")
        elif context.complexity_score > 0.8:
            recommendations.append("Use hybrid mode for comprehensive analysis of complex queries")

        # Data size recommendations
        data_size = context.data_shape[0] * context.data_shape[1]
        if data_size > 100000:
            recommendations.append("Large dataset detected - consider data sampling or parallel processing")

        # Capability recommendations
        if len(context.required_capabilities) > 3:
            recommendations.append("Multiple capabilities required - hybrid mode recommended for best results")

        # Agent-specific recommendations
        high_performing_agents = [agent for agent in relevant_agents if agent['success_rate'] > 0.9]
        if high_performing_agents:
            recommendations.append(
                f"High-performing agents available: {', '.join([a['agent_role'].value for a in high_performing_agents])}")

        return recommendations


# Factory function for easy initialization
async def create_hybrid_compliance_system() -> HybridComplianceOrchestrator:
    """Create and initialize the hybrid compliance system"""
    orchestrator = HybridComplianceOrchestrator()

    # Wait for initialization to complete
    await asyncio.sleep(1)

    return orchestrator


# Example usage and testing
async def example_usage():
    """Example usage of the hybrid system"""
    print("=== Hybrid RAG-Agentic Banking Compliance System ===")

    # Create system
    system = await create_hybrid_compliance_system()

    # Create sample data
    sample_data = pd.DataFrame({
        'Account_ID': [f'ACC{i:06d}' for i in range(100)],
        'Account_Type': ['Current', 'Savings', 'Fixed'] * 33 + ['Investment'],
        'Current_Balance': np.random.uniform(1000, 100000, 100),
        'Date_Last_Activity': pd.date_range('2020-01-01', periods=100, freq='D'),
        'Expected_Account_Dormant': np.random.choice(['yes', 'no'], 100, p=[0.3, 0.7])
    })

    # Example queries
    queries = [
        "How many accounts are dormant?",
        "What is the total balance of all dormant accounts?",
        "Show me compliance statistics for current accounts",
        "Analyze dormancy patterns by account type"
    ]

    print(f"Sample data shape: {sample_data.shape}")

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Processing query: '{query}'")

        # Get recommendations
        recommendations = await system.get_query_recommendations(query, sample_data)
        print(f"   Recommended mode: {recommendations.get('recommended_mode', 'unknown')}")
        print(f"   Complexity score: {recommendations.get('complexity_score', 0):.2f}")

        # Process with auto-selected mode
        result = await system.process_compliance_data(
            data=sample_data,
            query=query,
            session_id=f"example_session_{i}",
            processing_mode=ProcessingMode.AUTO_SELECT
        )

        print(f"   Result status: {result.status}")
        print(f"   Processing mode used: {result.processing_mode.value}")
        print(f"   Confidence score: {result.confidence_score:.2f}")
        print(f"   Processing time: {result.processing_time:.2f}s")

    # System status
    print(f"\n=== System Status ===")
    status = system.get_system_status()
    print(f"RAG agents: {status['rag_knowledge_base']['total_agents']}")
    print(f"Pandas queries executed: {status['pandas_agent']['performance']['total_queries']}")
    print(f"Sessions processed: {status['processing_history']['total_sessions']}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())