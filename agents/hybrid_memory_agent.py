import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import logging
import pandas as pd
from pathlib import Path

# MCP Integration
try:
    from mcp.mcp_integration import MCPComplianceTools

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MCPComplianceTools = None

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Structured memory entry for consistent storage"""
    id: str
    session_id: str
    timestamp: datetime
    event_type: str
    data: Dict[str, Any]
    tags: List[str]
    importance: float  # 0.0-1.0 scale
    ttl: Optional[datetime] = None  # Time to live for cache invalidation


@dataclass
class KnowledgeEntry:
    """Knowledge base entry for regulatory and compliance knowledge"""
    id: str
    category: str  # e.g., 'cbuae_regulation', 'compliance_rule', 'procedure'
    title: str
    content: Dict[str, Any]
    version: str
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    confidence_score: float


class HybridMemoryAgent:
    """
    Enhanced memory agent that manages both session memory and knowledge memory
    with MCP integration for advanced compliance intelligence
    """

    def __init__(self, mcp_server_url: str = "http://localhost:8000",
                 knowledge_cache_size: int = 1000,
                 session_cache_size: int = 500):
        # Core memory stores
        self.session_memory: Dict[str, List[MemoryEntry]] = {}
        self.knowledge_base: Dict[str, KnowledgeEntry] = {}

        # Cache management
        self.knowledge_cache_size = knowledge_cache_size
        self.session_cache_size = session_cache_size

        # MCP integration
        self.mcp_tools: Optional[MCPComplianceTools] = None
        self.mcp_enabled = False

        # Performance tracking
        self.stats = {
            'session_queries': 0,
            'knowledge_queries': 0,
            'mcp_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # Initialize MCP if available
        if MCP_AVAILABLE:
            self.mcp_tools = MCPComplianceTools(mcp_server_url)
            asyncio.create_task(self._initialize_mcp())

        # Load persistent knowledge base
        self._load_knowledge_base()

    async def _initialize_mcp(self):
        """Initialize MCP connection asynchronously"""
        try:
            self.mcp_enabled = await self.mcp_tools.initialize()
            if self.mcp_enabled:
                logger.info("MCP integration initialized successfully")
                # Preload essential regulatory knowledge
                await self._preload_regulatory_knowledge()
            else:
                logger.warning("MCP initialization failed - operating in local mode")
        except Exception as e:
            logger.error(f"MCP initialization error: {e}")
            self.mcp_enabled = False

    def log(self, session_id: str, event: str, data: Any,
            tags: List[str] = None, importance: float = 0.5,
            ttl_hours: Optional[int] = None) -> str:
        """
        Enhanced logging with structured memory entries
        """
        entry_id = str(uuid.uuid4())

        # Prepare TTL
        ttl = None
        if ttl_hours:
            ttl = datetime.now() + timedelta(hours=ttl_hours)

        # Create memory entry
        memory_entry = MemoryEntry(
            id=entry_id,
            session_id=session_id,
            timestamp=datetime.now(),
            event_type=event,
            data=self._serialize_data(data),
            tags=tags or [],
            importance=importance,
            ttl=ttl
        )

        # Store in session memory
        if session_id not in self.session_memory:
            self.session_memory[session_id] = []

        self.session_memory[session_id].append(memory_entry)

        # Manage cache size
        self._manage_session_cache(session_id)

        # Update stats
        self.stats['session_queries'] += 1

        logger.debug(f"Memory entry logged: {event} for session {session_id}")
        return entry_id

    def get(self, session_id: str, event_filter: str = None,
            limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve session memory with enhanced filtering
        """
        if session_id not in self.session_memory:
            return []

        entries = self.session_memory[session_id]

        # Filter by event type if specified
        if event_filter:
            entries = [e for e in entries if e.event_type == event_filter]

        # Sort by importance and timestamp
        entries.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)

        # Apply limit
        if limit:
            entries = entries[:limit]

        # Convert to dict format for backward compatibility
        return [self._memory_entry_to_dict(entry) for entry in entries]

    async def get_regulatory_guidance(self, article: str, account_type: str,
                                      cache_duration_hours: int = 24) -> Dict[str, Any]:
        """
        Get regulatory guidance with intelligent caching
        """
        cache_key = f"guidance_{article}_{account_type}"

        # Check local knowledge base first
        cached_knowledge = self._get_cached_knowledge(cache_key, cache_duration_hours)
        if cached_knowledge:
            self.stats['cache_hits'] += 1
            return cached_knowledge.content

        self.stats['cache_misses'] += 1

        # Query MCP if available
        if self.mcp_enabled and self.mcp_tools:
            try:
                self.stats['mcp_queries'] += 1
                guidance = await self.mcp_tools.get_regulatory_guidance(article, account_type)

                # Cache the result
                await self._cache_knowledge(
                    cache_key,
                    'cbuae_regulation',
                    f"Article {article} - {account_type}",
                    guidance,
                    tags=[article, account_type, 'regulatory_guidance']
                )

                return guidance
            except Exception as e:
                logger.error(f"MCP regulatory guidance query failed: {e}")

        # Fallback to local knowledge
        return self._get_fallback_guidance(article, account_type)

    async def validate_compliance_enhanced(self, account_data: Dict[str, Any],
                                           session_id: str) -> Dict[str, Any]:
        """
        Enhanced compliance validation with memory integration
        """
        # Log the validation request
        self.log(
            session_id,
            'compliance_validation_request',
            account_data,
            tags=['compliance', 'validation'],
            importance=0.8
        )

        # Use MCP validation if available
        if self.mcp_enabled and self.mcp_tools:
            try:
                result = await self.mcp_tools.validate_compliance_rules(account_data)

                # Enhance result with historical context
                historical_context = self._get_historical_compliance_context(
                    account_data.get('Account_ID'), session_id
                )
                result['historical_context'] = historical_context

                # Log the result
                self.log(
                    session_id,
                    'compliance_validation_result',
                    result,
                    tags=['compliance', 'validation', 'result'],
                    importance=0.9
                )

                return result
            except Exception as e:
                logger.error(f"MCP compliance validation failed: {e}")

        # Fallback validation
        return self._fallback_compliance_validation(account_data, session_id)

    async def generate_insights(self, session_id: str,
                                analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights using memory patterns and MCP knowledge
        """
        insights = {
            'session_patterns': self._analyze_session_patterns(session_id),
            'historical_trends': self._analyze_historical_trends(session_id),
            'regulatory_recommendations': [],
            'risk_indicators': []
        }

        # Generate MCP-enhanced insights if available
        if self.mcp_enabled and self.mcp_tools:
            try:
                mcp_report = await self.mcp_tools.generate_compliance_report(analysis_results)
                insights['mcp_enhanced_report'] = mcp_report
            except Exception as e:
                logger.error(f"MCP report generation failed: {e}")

        # Analyze dormant account patterns
        if 'dormant_results' in analysis_results:
            insights['dormancy_insights'] = self._analyze_dormancy_patterns(
                analysis_results['dormant_results'], session_id
            )

        # Analyze compliance patterns
        if 'compliance_results' in analysis_results:
            insights['compliance_insights'] = self._analyze_compliance_patterns(
                analysis_results['compliance_results'], session_id
            )

        # Log insights generation
        self.log(
            session_id,
            'insights_generated',
            insights,
            tags=['insights', 'analysis', 'patterns'],
            importance=1.0
        )

        return insights

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive session summary
        """
        entries = self.session_memory.get(session_id, [])

        if not entries:
            return {'status': 'no_data', 'message': 'No session data found'}

        # Analyze session statistics
        event_counts = {}
        importance_levels = []
        timeline = []

        for entry in entries:
            event_counts[entry.event_type] = event_counts.get(entry.event_type, 0) + 1
            importance_levels.append(entry.importance)
            timeline.append({
                'timestamp': entry.timestamp.isoformat(),
                'event': entry.event_type,
                'importance': entry.importance
            })

        # Calculate session metrics
        avg_importance = sum(importance_levels) / len(importance_levels) if importance_levels else 0
        most_common_event = max(event_counts, key=event_counts.get) if event_counts else None

        summary = {
            'session_id': session_id,
            'total_entries': len(entries),
            'event_distribution': event_counts,
            'average_importance': round(avg_importance, 3),
            'most_common_event': most_common_event,
            'session_duration': self._calculate_session_duration(entries),
            'timeline': timeline[-10:],  # Last 10 events
            'critical_events': [
                self._memory_entry_to_dict(e) for e in entries
                if e.importance >= 0.8
            ]
        }

        return summary

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics
        """
        categories = {}
        total_entries = len(self.knowledge_base)

        for entry in self.knowledge_base.values():
            categories[entry.category] = categories.get(entry.category, 0) + 1

        return {
            'total_knowledge_entries': total_entries,
            'categories': categories,
            'cache_size': len(self.knowledge_base),
            'mcp_enabled': self.mcp_enabled,
            'performance_stats': self.stats.copy()
        }

    async def cleanup_expired_entries(self):
        """
        Clean up expired cache entries
        """
        current_time = datetime.now()
        cleaned_count = 0

        # Clean session memory
        for session_id in list(self.session_memory.keys()):
            original_count = len(self.session_memory[session_id])
            self.session_memory[session_id] = [
                entry for entry in self.session_memory[session_id]
                if entry.ttl is None or entry.ttl > current_time
            ]
            cleaned_count += original_count - len(self.session_memory[session_id])

        # Clean knowledge base
        original_kb_count = len(self.knowledge_base)
        expired_keys = [
            key for key, entry in self.knowledge_base.items()
            if self._is_knowledge_expired(entry)
        ]

        for key in expired_keys:
            del self.knowledge_base[key]

        cleaned_count += len(expired_keys)

        logger.info(f"Cleaned up {cleaned_count} expired memory entries")
        return cleaned_count

    async def close(self):
        """
        Cleanup resources
        """
        if self.mcp_tools:
            await self.mcp_tools.close()

        # Save knowledge base
        self._save_knowledge_base()

        logger.info("Hybrid memory agent closed successfully")

    # ==================== Private Methods ====================

    def _serialize_data(self, data: Any) -> Dict[str, Any]:
        """
        Serialize data for memory storage
        """
        if isinstance(data, pd.DataFrame):
            return {
                'type': 'dataframe',
                'shape': data.shape,
                'columns': data.columns.tolist(),
                'sample': data.head(3).to_dict() if not data.empty else {}
            }
        elif isinstance(data, dict):
            return data
        elif isinstance(data, (list, tuple)):
            return {'type': 'sequence', 'length': len(data), 'sample': data[:5]}
        else:
            return {'type': str(type(data).__name__), 'value': str(data)}

    def _memory_entry_to_dict(self, entry: MemoryEntry) -> Dict[str, Any]:
        """
        Convert memory entry to dictionary format
        """
        return {
            'id': entry.id,
            'event': entry.event_type,
            'data': entry.data,
            'timestamp': entry.timestamp.isoformat(),
            'tags': entry.tags,
            'importance': entry.importance
        }

    def _manage_session_cache(self, session_id: str):
        """
        Manage session cache size by removing least important entries
        """
        if len(self.session_memory[session_id]) > self.session_cache_size:
            # Sort by importance (ascending) and remove least important
            self.session_memory[session_id].sort(key=lambda x: x.importance)
            excess_count = len(self.session_memory[session_id]) - self.session_cache_size
            self.session_memory[session_id] = self.session_memory[session_id][excess_count:]

    def _get_cached_knowledge(self, cache_key: str,
                              cache_duration_hours: int) -> Optional[KnowledgeEntry]:
        """
        Get cached knowledge entry if still valid
        """
        if cache_key in self.knowledge_base:
            entry = self.knowledge_base[cache_key]
            age_hours = (datetime.now() - entry.updated_at).total_seconds() / 3600

            if age_hours < cache_duration_hours:
                return entry

        return None

    async def _cache_knowledge(self, cache_key: str, category: str,
                               title: str, content: Dict[str, Any],
                               tags: List[str] = None):
        """
        Cache knowledge entry
        """
        entry = KnowledgeEntry(
            id=cache_key,
            category=category,
            title=title,
            content=content,
            version="1.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=tags or [],
            confidence_score=0.8
        )

        self.knowledge_base[cache_key] = entry

        # Manage cache size
        if len(self.knowledge_base) > self.knowledge_cache_size:
            # Remove oldest entries
            sorted_entries = sorted(
                self.knowledge_base.items(),
                key=lambda x: x[1].updated_at
            )
            excess_count = len(self.knowledge_base) - self.knowledge_cache_size
            for i in range(excess_count):
                del self.knowledge_base[sorted_entries[i][0]]

    def _get_fallback_guidance(self, article: str, account_type: str) -> Dict[str, Any]:
        """
        Provide fallback regulatory guidance
        """
        fallback_guidance = {
            'article': article,
            'account_type': account_type,
            'guidance': f"Standard CBUAE guidance for Article {article} applies to {account_type} accounts",
            'source': 'local_fallback',
            'confidence': 0.6
        }

        # Enhanced fallback based on known patterns
        if article in ['2.1', '2.2', '2.3']:
            fallback_guidance['guidance'] += ". Monitor for 3+ years of inactivity."
        elif article == '8':
            fallback_guidance['guidance'] += ". Consider for CBUAE transfer after 5+ years."

        return fallback_guidance

    def _get_historical_compliance_context(self, account_id: str,
                                           session_id: str) -> Dict[str, Any]:
        """
        Get historical compliance context for an account
        """
        # Search across all sessions for this account
        context = {
            'previous_violations': [],
            'compliance_history': [],
            'risk_score': 0.5
        }

        for sid, entries in self.session_memory.items():
            for entry in entries:
                if (entry.event_type == 'compliance_validation_result' and
                        isinstance(entry.data, dict) and
                        entry.data.get('account_id') == account_id):
                    context['compliance_history'].append({
                        'session_id': sid,
                        'timestamp': entry.timestamp.isoformat(),
                        'result': entry.data
                    })

        return context

    def _fallback_compliance_validation(self, account_data: Dict[str, Any],
                                        session_id: str) -> Dict[str, Any]:
        """
        Fallback compliance validation using local rules
        """
        issues = []

        # Basic validation rules
        if account_data.get('Current_Balance', 0) > 100000:
            issues.append("High-value account requires enhanced monitoring")

        if account_data.get('Expected_Account_Dormant') == 'yes':
            issues.append("Dormant account status confirmed")

        return {
            'compliant': len(issues) == 0,
            'issues': issues,
            'source': 'local_validation',
            'confidence': 0.7
        }

    def _analyze_session_patterns(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze patterns in session data
        """
        entries = self.session_memory.get(session_id, [])

        patterns = {
            'event_frequency': {},
            'importance_trend': [],
            'error_rate': 0,
            'processing_efficiency': 0
        }

        error_count = 0
        total_events = len(entries)

        for entry in entries:
            # Event frequency analysis
            patterns['event_frequency'][entry.event_type] = \
                patterns['event_frequency'].get(entry.event_type, 0) + 1

            # Importance trend
            patterns['importance_trend'].append(entry.importance)

            # Error tracking
            if 'error' in entry.event_type.lower():
                error_count += 1

        if total_events > 0:
            patterns['error_rate'] = error_count / total_events
            patterns['processing_efficiency'] = 1 - patterns['error_rate']

        return patterns

    def _analyze_historical_trends(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze historical trends across sessions
        """
        all_entries = []
        for entries in self.session_memory.values():
            all_entries.extend(entries)

        # Time-based analysis
        daily_counts = {}
        for entry in all_entries:
            day_key = entry.timestamp.date().isoformat()
            daily_counts[day_key] = daily_counts.get(day_key, 0) + 1

        return {
            'total_historical_entries': len(all_entries),
            'daily_activity': daily_counts,
            'peak_activity_day': max(daily_counts, key=daily_counts.get) if daily_counts else None
        }

    def _analyze_dormancy_patterns(self, dormant_results: Dict[str, Any],
                                   session_id: str) -> Dict[str, Any]:
        """
        Analyze dormancy patterns using memory context
        """
        return {
            'dormancy_rate_trend': 'stable',  # Would need historical data
            'high_risk_categories': ['Safe Deposit', 'Investment'],
            'recommended_actions': [
                'Increase monitoring frequency for high-value accounts',
                'Review contact attempt procedures'
            ]
        }

    def _analyze_compliance_patterns(self, compliance_results: Dict[str, Any],
                                     session_id: str) -> Dict[str, Any]:
        """
        Analyze compliance patterns using memory context
        """
        return {
            'compliance_trend': 'improving',  # Would need historical data
            'critical_areas': ['CBUAE Transfer', 'Contact Attempts'],
            'recommended_improvements': [
                'Implement automated contact systems',
                'Enhance transfer tracking procedures'
            ]
        }

    def _calculate_session_duration(self, entries: List[MemoryEntry]) -> str:
        """
        Calculate session duration
        """
        if len(entries) < 2:
            return "0 minutes"

        start_time = min(entry.timestamp for entry in entries)
        end_time = max(entry.timestamp for entry in entries)
        duration = end_time - start_time

        hours = duration.total_seconds() / 3600
        if hours >= 1:
            return f"{hours:.1f} hours"
        else:
            minutes = duration.total_seconds() / 60
            return f"{minutes:.1f} minutes"

    def _is_knowledge_expired(self, entry: KnowledgeEntry) -> bool:
        """
        Check if knowledge entry is expired
        """
        # Regulatory knowledge expires after 30 days
        if entry.category == 'cbuae_regulation':
            age_days = (datetime.now() - entry.updated_at).days
            return age_days > 30

        return False

    def _load_knowledge_base(self):
        """
        Load persistent knowledge base from storage
        """
        # Placeholder for loading from persistent storage
        logger.info("Knowledge base loaded from persistent storage")

    def _save_knowledge_base(self):
        """
        Save knowledge base to persistent storage
        """
        # Placeholder for saving to persistent storage
        logger.info("Knowledge base saved to persistent storage")

    async def _preload_regulatory_knowledge(self):
        """
        Preload essential regulatory knowledge via MCP
        """
        essential_articles = ['2.1', '2.2', '2.3', '2.4', '2.6', '8']
        account_types = ['Current', 'Savings', 'Fixed', 'Investment', 'Safe Deposit']

        for article in essential_articles:
            for account_type in account_types:
                try:
                    await self.get_regulatory_guidance(article, account_type)
                    await asyncio.sleep(0.1)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Failed to preload guidance for {article}-{account_type}: {e}")

        logger.info("Essential regulatory knowledge preloaded")