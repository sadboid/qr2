from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class ResearchQuestion(Base):
    """Stores research questions"""
    __tablename__ = "research_questions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    question = Column(Text, nullable=False)
    domain = Column(String, nullable=False)  # 'startup' or 'enterprise'
    keywords = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    papers = relationship("Paper", back_populates="research_question")


class Paper(Base):
    """Stores generated research papers"""
    __tablename__ = "papers"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    research_question_id = Column(String, ForeignKey("research_questions.id"))
    title = Column(String, nullable=False)
    abstract = Column(Text, nullable=True)
    content_markdown = Column(Text, nullable=True)
    content_latex = Column(Text, nullable=True)

    # Quality Metrics
    novelty_score = Column(Float, nullable=True)
    rigor_score = Column(Float, nullable=True)
    citation_count = Column(Integer, nullable=True)
    h_index_avg = Column(Float, nullable=True)
    peer_review_score = Column(Float, nullable=True)

    # Status
    status = Column(String, default="draft")  # draft, accepted, rejected, needs_revision
    quality_feedback = Column(Text, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    research_question = relationship("ResearchQuestion", back_populates="papers")
    citations = relationship("Citation", back_populates="paper")


class Citation(Base):
    """Stores paper citations"""
    __tablename__ = "citations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    paper_id = Column(String, ForeignKey("papers.id"))
    external_id = Column(String, nullable=True)  # arXiv ID, Semantic Scholar ID
    title = Column(String, nullable=True)
    authors = Column(JSON, nullable=True)
    year = Column(Integer, nullable=True)
    venue = Column(String, nullable=True)
    h_index = Column(Integer, nullable=True)
    url = Column(String, nullable=True)
    abstract = Column(Text, nullable=True)

    paper = relationship("Paper", back_populates="citations")


class SearchCache(Base):
    """Cache for literature search results to avoid redundant API calls"""
    __tablename__ = "search_cache"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query = Column(String, nullable=False, unique=True)
    results = Column(JSON, nullable=False)  # Cached API response
    source = Column(String, nullable=False)  # 'semantic_scholar', 'arxiv'
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # For cache expiration


class PaperEmbedding(Base):
    """Stores vector embeddings for papers (for similarity search)"""
    __tablename__ = "paper_embeddings"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    paper_id = Column(String, ForeignKey("papers.id"))
    embedding_id = Column(String, nullable=False)  # Qdrant collection ID
    embedding_model = Column(String, default="all-MiniLM-L6-v2")
    created_at = Column(DateTime, default=datetime.utcnow)
