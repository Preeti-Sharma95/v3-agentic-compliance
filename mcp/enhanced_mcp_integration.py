# mcp/enhanced_mcp_integration.py
import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import aiohttp
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MCPRequest:
    """Structured MCP request with memory context"""
    tool: str
    arguments: Dict[str, Any]
    session_id: str
    context: Optional[Dict[str, Any]] = None
    priority: str = "normal"  # low, normal, high, critical


@dataclass
class MCPResponse:
    """Structured MCP response with metadata"""
    success: bool
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None
    cached: bool = False
    response_time_ms: int = 0


class EnhancedMCPComplianceTools:
    """
    Enhanced MCP integration with memory-aware compliance tools
    """

    def __init__(self, server_url: str = "http://localhost:8000",
                 timeout: int = 30, max_retries: int = 3):
        self.server_url = server_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        self.initialized = False

        # Performance tracking
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'cached_responses': 0
        }

        # Request queue for priority handling
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.processing_requests = False

    async def initialize(self) -> bool:
        """Initialize MCP client with enhanced error handling"""
        try:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                keepalive_timeout=30
            )

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'Banking-Compliance-Agent/1.0'
                }
            )

            # Test connection with health check
            health_response = await self._health_check()
            if health_response.success:
                self.initialized = True
                logger.info(f"MCP client initialized successfully - Server: {self.server_url}")

                # Start request processor
                asyncio.create_task(self._process_request_queue())

                return True
            else:
                logger.error(f"MCP server health check failed: {health_response.error}")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize MCP session: {e}")
            self.initialized = False
            return False

    async def close(self):
        """Close MCP session with cleanup"""
        self.processing_requests = False

        if self.session:
            await self.session.close()

        logger.info("Enhanced MCP client closed successfully")

    async def get_regulatory_guidance_enhanced(self,
                                               article: str,
                                               account_type: str,
                                               session_id: str,
                                               context: Dict[str, Any] = None) -> MCPResponse:
        """
        Get regulatory guidance with enhanced context and memory integration
        """
        request = MCPRequest(
            tool="get_regulatory_guidance",
            arguments={
                "article": article,
                "account_type": account_type,
                "language": "en",
                "include_examples": True,
                "include_penalties": True
            },
            session_id=session_id,
            context=context or {},
            priority="high"
        )

        return await self._execute_request(request)

    async def validate_compliance_rules_enhanced(self,
                                                 account_data: Dict[str, Any],
                                                 session_id: str,
                                                 validation_level: str = "comprehensive") -> MCPResponse:
        """
        Enhanced compliance validation with multiple validation levels
        """
        request = MCPRequest(
            tool="validate_compliance",
            arguments={
                "account_data": account_data,
                "validation_level": validation_level,
                "include_recommendations": True,
                "check_historical_patterns": True
            },
            session_id=session_id,
            priority="critical"
        )

        return await self._execute_request(request)

    async def generate_compliance_report_enhanced(self,
                                                  analysis_results: Dict[str, Any],
                                                  session_id: str,
                                                  report_type: str = "comprehensive") -> MCPResponse:
        """
        Generate enhanced compliance report with memory context
        """
        request = MCPRequest(
            tool="generate_report",
            arguments={
                "analysis_results": analysis_results,
                "report_type": report_type,
                "include_visualizations": True,
                "include_action_items": True,
                "regulatory_framework": "CBUAE"
            },
            session_id=session_id,
            priority="normal"
        )

        return await self._execute_request(request)

    async def query_knowledge_base(self,
                                   query: str,
                                   session_id: str,
                                   category: str = "all") -> MCPResponse:
        """
        Query the MCP knowledge base for compliance information
        """
        request = MCPRequest(
            tool="query_knowledge",
            arguments={
                "query": query,
                "category": category,
                "max_results": 10,
                "include_confidence": True
            },
            session_id=session_id,
            priority="normal"
        )

        return await self._execute_request(request)

    async def analyze_dormancy_patterns(self,
                                        dormant_accounts: List[Dict[str, Any]],
                                        session_id: str) -> MCPResponse:
        """
        Analyze dormancy patterns using MCP intelligence
        """
        request = MCPRequest(
            tool="analyze_dormancy",
            arguments={
                "dormant_accounts": dormant_accounts,
                "analysis_type": "pattern_detection",
                "include_predictions": True,
                "risk_assessment": True
            },
            session_id=session_id,
            priority="high"
        )

        return await self._execute_request(request)

    async def get_compliance_recommendations(self,
                                             compliance_issues: Dict[str, Any],
                                             session_id: str) -> MCPResponse:
        """
        Get AI-powered compliance recommendations
        """
        request = MCPRequest(
            tool="get_recommendations",
            arguments={
                "compliance_issues": compliance_issues,
                "urgency_level": "auto_detect",
                "include_timeline": True,
                "regulatory_context": "UAE_CBUAE"
            },
            session_id=session_id,
            priority="high"
        )

        return await self._execute_request(request)

    async def validate_transaction_patterns(self,
                                            transactions: List[Dict[str, Any]],
                                            session_id: str) -> MCPResponse:
        """
        Validate transaction patterns for compliance anomalies
        """
        request = MCPRequest(
            tool="validate_transactions",
            arguments={
                "transactions": transactions,
                "detection_rules": ["aml", "dormancy", "suspicious_activity"],
                "confidence_threshold": 0.8
            },
            session_id=session_id,
            priority="critical"
        )

        return await self._execute_request(request)

    async def get_regulatory_updates(self,
                                     last_check: datetime,
                                     session_id: str) -> MCPResponse:
        """
        Get latest regulatory updates from MCP knowledge base
        """
        request = MCPRequest(
            tool="get_updates",
            arguments={
                "last_check": last_check.isoformat(),
                "update_types": ["regulations", "guidelines", "circulars"],
                "jurisdiction": "UAE"
            },
            session_id=session_id,
            priority="low"
        )

        return await self._execute_request(request)

    async def batch_compliance_check(self,
                                     accounts_batch: List[Dict[str, Any]],
                                     session_id: str,
                                     batch_size: int = 100) -> List[MCPResponse]:
        """
        Perform batch compliance checking with optimized processing
        """
        responses = []

        # Split into chunks for processing
        for i in range(0, len(accounts_batch), batch_size):
            batch = accounts_batch[i:i + batch_size]

            request = MCPRequest(
                tool="batch_validate",
                arguments={
                    "accounts_batch": batch,
                    "batch_id": f"{session_id}_batch_{i // batch_size}",
                    "parallel_processing": True
                },
                session_id=session_id,
                priority="normal"
            )

            response = await self._execute_request(request)
            responses.append(response)

            # Brief pause between batches to avoid overwhelming the server
            await asyncio.sleep(0.1)

        return responses

    # ==================== Private Methods ====================

    async def _execute_request(self, request: MCPRequest) -> MCPResponse:
        """
        Execute MCP request with retry logic and performance tracking
        """
        if not self.initialized:
            return MCPResponse(
                success=False,
                data={},
                metadata={"error_type": "client_not_initialized"},
                error="MCP client not initialized"
            )

        start_time = datetime.now()

        for attempt in range(self.max_retries):
            try:
                # Add request to queue if high priority
                if request.priority in ["critical", "high"]:
                    response = await self._send_request_direct(request)
                else:
                    await self.request_queue.put(request)
                    response = await self._wait_for_response(request)

                # Calculate response time
                response_time = int((datetime.now() - start_time).total_seconds() * 1000)
                response.response_time_ms = response_time

                # Update statistics
                self._update_stats(response)

                return response

            except asyncio.TimeoutError:
                logger.warning(f"MCP request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    return self._create_error_response("Request timeout", request)
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                logger.error(f"MCP request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return self._create_error_response(str(e), request)
                await asyncio.sleep(2 ** attempt)

        return self._create_error_response("Max retries exceeded", request)

    async def _send_request_direct(self, request: MCPRequest) -> MCPResponse:
        """
        Send request directly to MCP server
        """
        payload = {
            "tool": request.tool,
            "arguments": request.arguments,
            "metadata": {
                "session_id": request.session_id,
                "timestamp": datetime.now().isoformat(),
                "priority": request.priority,
                "context": request.context
            }
        }

        async with self.session.post(f"{self.server_url}/tools", json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return MCPResponse(
                    success=True,
                    data=data.get("result", {}),
                    metadata=data.get("metadata", {}),
                    cached=data.get("cached", False)
                )
            else:
                error_text = await response.text()
                return MCPResponse(
                    success=False,
                    data={},
                    metadata={"http_status": response.status},
                    error=f"HTTP {response.status}: {error_text}"
                )

    async def _wait_for_response(self, request: MCPRequest) -> MCPResponse:
        """
        Wait for queued request response
        """
        # For now, process immediately (queue processing can be enhanced)
        return await self._send_request_direct(request)

    async def _process_request_queue(self):
        """
        Process requests from the queue with priority handling
        """
        self.processing_requests = True

        while self.processing_requests:
            try:
                # Process requests from queue
                if not self.request_queue.empty():
                    request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=1.0
                    )

                    response = await self._send_request_direct(request)
                    # Store response for retrieval (implementation depends on architecture)

                else:
                    await asyncio.sleep(0.1)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Request queue processing error: {e}")
                await asyncio.sleep(1)

    async def _health_check(self) -> MCPResponse:
        """
        Perform health check on MCP server
        """
        try:
            async with self.session.get(f"{self.server_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return MCPResponse(
                        success=True,
                        data=data,
                        metadata={"health_check": True}
                    )
                else:
                    return MCPResponse(
                        success=False,
                        data={},
                        metadata={"http_status": response.status},
                        error=f"Health check failed with status {response.status}"
                    )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                metadata={"health_check": True},
                error=f"Health check error: {str(e)}"
            )

    def _create_error_response(self, error_msg: str, request: MCPRequest) -> MCPResponse:
        """
        Create standardized error response
        """
        return MCPResponse(
            success=False,
            data={},
            metadata={
                "tool": request.tool,
                "session_id": request.session_id,
                "error_type": "execution_failed"
            },
            error=error_msg
        )

    def _update_stats(self, response: MCPResponse):
        """
        Update performance statistics
        """
        self.request_stats['total_requests'] += 1

        if response.success:
            self.request_stats['successful_requests'] += 1
        else:
            self.request_stats['failed_requests'] += 1

        if response.cached:
            self.request_stats['cached_responses'] += 1

        # Update average response time
        total_requests = self.request_stats['total_requests']
        current_avg = self.request_stats['average_response_time']
        new_avg = ((current_avg * (total_requests - 1)) + response.response_time_ms) / total_requests
        self.request_stats['average_response_time'] = round(new_avg, 2)

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        """
        stats = self.request_stats.copy()

        if stats['total_requests'] > 0:
            stats['success_rate'] = round(
                (stats['successful_requests'] / stats['total_requests']) * 100, 2
            )
            stats['cache_hit_rate'] = round(
                (stats['cached_responses'] / stats['total_requests']) * 100, 2
            )
        else:
            stats['success_rate'] = 0
            stats['cache_hit_rate'] = 0

        return stats


class MCPKnowledgeManager:
    """
    Specialized MCP manager for knowledge base operations
    """

    def __init__(self, mcp_tools: EnhancedMCPComplianceTools):
        self.mcp_tools = mcp_tools
        self.knowledge_cache = {}
        self.cache_ttl = timedelta(hours=24)

    async def get_regulatory_article(self, article_id: str, session_id: str) -> Dict[str, Any]:
        """
        Get complete regulatory article content
        """
        cache_key = f"article_{article_id}"

        # Check cache first
        if self._is_cached_valid(cache_key):
            return self.knowledge_cache[cache_key]['data']

        # Query MCP
        response = await self.mcp_tools.query_knowledge_base(
            query=f"regulatory article {article_id}",
            session_id=session_id,
            category="regulations"
        )

        if response.success:
            # Cache the result
            self.knowledge_cache[cache_key] = {
                'data': response.data,
                'timestamp': datetime.now()
            }
            return response.data

        return {"error": "Failed to retrieve article", "article_id": article_id}

    async def search_compliance_procedures(self, query: str, session_id: str) -> List[Dict[str, Any]]:
        """
        Search for compliance procedures
        """
        response = await self.mcp_tools.query_knowledge_base(
            query=query,
            session_id=session_id,
            category="procedures"
        )

        if response.success:
            return response.data.get('results', [])

        return []

    async def get_penalty_information(self, violation_type: str, session_id: str) -> Dict[str, Any]:
        """
        Get penalty information for compliance violations
        """
        response = await self.mcp_tools.query_knowledge_base(
            query=f"penalty {violation_type}",
            session_id=session_id,
            category="penalties"
        )

        if response.success:
            return response.data

        return {"error": "Penalty information not found", "violation_type": violation_type}

    def _is_cached_valid(self, cache_key: str) -> bool:
        """
        Check if cached entry is still valid
        """
        if cache_key not in self.knowledge_cache:
            return False

        cached_time = self.knowledge_cache[cache_key]['timestamp']
        return datetime.now() - cached_time < self.cache_ttl

    def clear_cache(self):
        """
        Clear the knowledge cache
        """
        self.knowledge_cache.clear()
        logger.info("Knowledge cache cleared")


# Factory function for creating enhanced MCP tools
async def create_enhanced_mcp_tools(server_url: str = "http://localhost:8000") -> Optional[EnhancedMCPComplianceTools]:
    """
    Factory function to create and initialize enhanced MCP tools
    """
    tools = EnhancedMCPComplianceTools(server_url)

    if await tools.initialize():
        return tools
    else:
        logger.warning("Failed to initialize MCP tools - falling back to local mode")
        return None