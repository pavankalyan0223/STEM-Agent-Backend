"""
Research Graph Generator
Builds an interactive knowledge graph from research papers using:
- KeyBERT ONLY for keyword extraction (no other keyword extraction methods)
- SentenceTransformer for embeddings (topic similarity only, not keyword extraction)
- NetworkX for graph structure
"""

import os
import json
import numpy as np
import re
from typing import Dict, List, Any, Set, Tuple
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optional imports - make them optional so server can start without them
try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    KeyBERT = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

RESEARCH_GRAPH_DIR = "data/research_graphs"
os.makedirs(RESEARCH_GRAPH_DIR, exist_ok=True)

# Initialize models (reuse embedder from rag_setup if available)
kw_model = None
embedder = None

if KEYBERT_AVAILABLE and SENTENCE_TRANSFORMER_AVAILABLE:
    print("🔧 Initializing models...")
    try:
        kw_model = KeyBERT()
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"⚠️  Error loading models: {e}")
        kw_model = None
        embedder = None
else:
    missing = []
    if not KEYBERT_AVAILABLE:
        missing.append("keybert")
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        missing.append("sentence-transformers")
    if not NETWORKX_AVAILABLE:
        missing.append("networkx")
    print(f"⚠️  Missing packages: {', '.join(missing)}. Install with: pip install {' '.join(missing)}")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    # Clean up common PDF extraction artifacts
    # Normalize whitespace but preserve structure
    text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces and tabs
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove excessive newlines
    # Fix common PDF extraction issues: broken words across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix hyphenated words split across lines
    return text


def is_valid_keyword(keyword: str, source_text: str) -> bool:
    """
    Validate that a keyword actually appears in the source text.
    Filters out broken words and non-existent keywords.
    """
    if not keyword or len(keyword.strip()) < 2:
        return False
    
    keyword_clean = keyword.lower().strip()
    source_lower = source_text.lower()
    
    # Check if keyword appears as a whole phrase in the text
    # Use word boundaries to avoid partial matches
    words = keyword_clean.split()
    
    # For single words, check if it appears as a complete word
    if len(words) == 1:
        # Check if word appears with word boundaries
        pattern = r'\b' + re.escape(keyword_clean) + r'\b'
        if re.search(pattern, source_lower):
            return True
    
    # For multi-word phrases, check if all words appear together
    elif len(words) > 1:
        # Check if phrase appears in text (allowing for some spacing variations)
        phrase_pattern = r'\b' + r'\s+'.join([re.escape(w) for w in words]) + r'\b'
        if re.search(phrase_pattern, source_lower):
            return True
    
    # Also check if keyword appears without strict word boundaries (for hyphenated words, etc.)
    if keyword_clean in source_lower:
        return True
    
    return False


def clean_keyword(keyword: str) -> str:
    """Clean and normalize a keyword."""
    # Remove extra whitespace
    keyword = ' '.join(keyword.split())
    # Remove leading/trailing punctuation except hyphens
    keyword = keyword.strip('.,;:()[]{}"\'').strip()
    return keyword


def extract_keywords_keybert(text: str, top_n: int = 15) -> List[Tuple[str, float]]:
    """
    Extract keywords from text using KeyBERT ONLY.
    This is the only keyword extraction method used in this codebase.
    Validates that keywords actually appear in the source text.
    Returns list of (keyword, score) tuples.
    """
    if kw_model is None:
        return []
    
    try:
        # Extract keywords with scores
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),  # Single words and 2-word phrases
            stop_words='english',
            top_n=top_n * 2,  # Extract more to filter invalid ones
            use_mmr=True,  # Maximal Marginal Relevance for diversity
            diversity=0.5
        )
        
        # Filter and validate keywords
        valid_keywords = []
        seen_keywords = set()
        
        for keyword, score in keywords:
            # Clean the keyword
            keyword_clean = clean_keyword(keyword)
            
            # Skip if empty or too short
            if not keyword_clean or len(keyword_clean) < 2:
                continue
            
            # Skip if contains invalid characters (broken words) - only allow alphanumeric, spaces, and hyphens
            if re.search(r'[^\w\s\-]', keyword_clean):
                continue
            
            # Skip if it's a broken word (contains numbers in middle, or weird patterns)
            if re.search(r'\d+[a-zA-Z]|[a-zA-Z]\d+', keyword_clean):
                # Allow if it's a valid scientific term (like "2D", "3D", etc.)
                if not re.match(r'^\d+[a-zA-Z]$', keyword_clean):
                    continue
            
            # Validate that keyword actually appears in source text
            if not is_valid_keyword(keyword_clean, text):
                continue
            
            # Check for duplicates (case-insensitive)
            keyword_lower = keyword_clean.lower()
            if keyword_lower in seen_keywords:
                continue
            
            seen_keywords.add(keyword_lower)
            valid_keywords.append((keyword_clean, score))
            
            # Stop when we have enough valid keywords
            if len(valid_keywords) >= top_n:
                break
        
        return valid_keywords
    except Exception as e:
        print(f"  ⚠️  KeyBERT error: {e}")
        return []


def extract_topics_from_paper(text: str, paper_title: str) -> List[Dict[str, Any]]:
    """
    Extract topics/keywords from paper using KeyBERT ONLY.
    Processes entire paper text and sections using KeyBERT's extract_keywords method.
    Validates that all keywords actually appear in the source text.
    """
    if kw_model is None:
        return []
    
    # Get keywords from full text (already validated in extract_keywords_keybert)
    keywords_with_scores = extract_keywords_keybert(text, top_n=20)
    
    # Also extract from sections for better coverage
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=300, length_function=len
    )
    sections = splitter.split_text(text)
    
    topics = []
    seen_topics = set()
    
    # Extract from sections
    for i, section in enumerate(sections[:15]):  # Process up to 15 sections
        section_keywords = extract_keywords_keybert(section, top_n=5)
        for keyword, score in section_keywords:
            keyword_clean = clean_keyword(keyword)
            keyword_lower = keyword_clean.lower().strip()
            
            # Additional validation: ensure keyword appears in section
            if not is_valid_keyword(keyword_clean, section):
                continue
            
            if keyword_lower and keyword_lower not in seen_topics and len(keyword_lower) > 3:
                seen_topics.add(keyword_lower)
                topics.append({
                    "name": keyword_clean,
                    "score": float(score),
                    "description": f"Extracted from section {i+1}"
                })
    
    # Add top keywords from full text (already validated)
    for keyword, score in keywords_with_scores[:15]:
        keyword_clean = clean_keyword(keyword)
        keyword_lower = keyword_clean.lower().strip()
        
        if keyword_lower not in seen_topics:
            seen_topics.add(keyword_lower)
            topics.append({
                "name": keyword_clean,
                "score": float(score),
                "description": f"Main keyword from {paper_title}"
            })
    
    return topics


def extract_paper_metadata(text: str, filename: str) -> Dict[str, Any]:
    """Extract basic metadata from paper text."""
    lines = text.split('\n')[:50]
    
    title = filename.replace('.pdf', '').replace('_', ' ')
    
    # Look for abstract section
    abstract = ""
    abstract_started = False
    for i, line in enumerate(lines):
        if 'abstract' in line.lower() and len(line) < 20:
            abstract_started = True
            continue
        if abstract_started and line.strip():
            abstract += line.strip() + " "
            if len(abstract) > 500:
                break
    
    return {
        "title": title,
        "abstract": abstract[:500] if abstract else "No abstract found.",
        "filename": filename
    }


def build_research_graph(pdf_dir: str = "data/") -> Dict[str, Any]:
    """
    Build a knowledge graph from all PDFs using KeyBERT, embeddings, and NetworkX.
    
    Graph structure:
    - Nodes: papers and topics/keywords
    - Edges: paper->topic (contains), topic->topic (similarity based on embeddings)
    """
    if not KEYBERT_AVAILABLE or not SENTENCE_TRANSFORMER_AVAILABLE or not NETWORKX_AVAILABLE:
        missing = []
        if not KEYBERT_AVAILABLE:
            missing.append("keybert")
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            missing.append("sentence-transformers")
        if not NETWORKX_AVAILABLE:
            missing.append("networkx")
        return {
            "error": f"Missing required packages: {', '.join(missing)}. Install with: pip install {' '.join(missing)}"
        }
    
    if kw_model is None or embedder is None:
        return {"error": "Models not initialized. Please check KeyBERT and SentenceTransformer installation."}
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    
    if not pdf_files:
        return {"error": "No PDF files found in data directory"}
    
    print(f"📚 Processing {len(pdf_files)} research papers...")
    
    # Process each paper
    papers = []
    all_topics = []
    
    for idx, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"\n📄 [{idx+1}/{len(pdf_files)}] Processing: {pdf_file}")
        
        try:
            text = extract_text_from_pdf(pdf_path)
            if not text.strip():
                print(f"  ⚠️  Could not extract text from {pdf_file}")
                continue
            
            # Extract metadata
            metadata = extract_paper_metadata(text, pdf_file)
            
            # Extract topics using KeyBERT
            print(f"  🔍 Extracting topics with KeyBERT...")
            topics = extract_topics_from_paper(text, metadata['title'])
            
            # Get keywords for paper
            keywords_with_scores = extract_keywords_keybert(text, top_n=15)
            keywords = [kw[0] for kw in keywords_with_scores]
            
            # Create paper node
            paper_node = {
                "id": f"paper_{idx}",
                "type": "paper",
                "label": metadata['title'],
                "title": metadata['title'],
                "abstract": metadata['abstract'],
                "filename": pdf_file,
                "topics_count": len(topics),
                "keywords": keywords
            }
            
            papers.append(paper_node)
            all_topics.append(topics)
            
            print(f"  ✅ Found {len(topics)} topics")
            
        except Exception as e:
            print(f"  ❌ Error processing {pdf_file}: {e}")
            continue
    
    if not papers:
        return {"error": "No papers were successfully processed"}
    
    # Build NetworkX graph
    G = nx.Graph()
    
    # Add paper nodes
    for paper in papers:
        G.add_node(paper['id'], **paper, node_type="paper")
    
    # Build topic nodes with better deduplication across papers
    topic_nodes = {}
    topic_name_to_id = {}  # Map normalized topic name to canonical topic_id
    topic_id_to_name = {}  # Map topic_id to topic name
    topic_embeddings = {}  # Store embeddings for topics
    
    for paper_idx, topics in enumerate(all_topics):
        paper_id = f"paper_{paper_idx}"
        
        for topic in topics:
            topic_name = topic['name']
            topic_name_normalized = topic_name.lower().strip()
            
            # Check if we've seen this topic before (across papers)
            if topic_name_normalized in topic_name_to_id:
                # Use existing topic ID
                topic_id = topic_name_to_id[topic_name_normalized]
                # Add paper to existing topic
                if paper_id not in topic_nodes[topic_id]["papers"]:
                    topic_nodes[topic_id]["papers"].append(paper_id)
            else:
                # Create new topic node
                topic_id = f"topic_{len(topic_nodes)}_{topic_name_normalized.replace(' ', '_').replace('-', '_')}"
                topic_name_to_id[topic_name_normalized] = topic_id
                topic_id_to_name[topic_id] = topic_name
                
                topic_nodes[topic_id] = {
                    "id": topic_id,
                    "type": "topic",
                    "label": topic_name,
                    "name": topic_name,
                    "description": topic.get('description', ''),
                    "score": topic.get('score', 0.0),
                    "papers": [paper_id]
                }
                G.add_node(topic_id, **topic_nodes[topic_id], node_type="topic")
                
                # Create embedding for this topic (only once per unique topic)
                if embedder:
                    topic_embedding = embedder.encode(topic_name)
                    topic_embeddings[topic_id] = topic_embedding
            
            # Create paper->topic edge
            if not G.has_edge(paper_id, topic_id):
                G.add_edge(paper_id, topic_id, type="contains_topic", weight=1.0)
    
    # Find similar topics using embeddings and name matching
    print(f"\n🔗 Finding topic similarities using embeddings...")
    topic_ids = list(topic_embeddings.keys())
    similarity_edges = []
    similarity_threshold = 0.4  # Lower threshold for more connections
    
    print(f"  Comparing {len(topic_ids)} unique topics...")
    
    for i, topic_id1 in enumerate(topic_ids):
        topic_name1 = topic_id_to_name.get(topic_id1, "").lower()
        emb1 = topic_embeddings[topic_id1]
        
        for topic_id2 in topic_ids[i+1:]:
            topic_name2 = topic_id_to_name.get(topic_id2, "").lower()
            
            # First check name similarity (exact or partial match)
            name_similarity = 0.0
            if topic_name1 == topic_name2:
                name_similarity = 1.0
            elif topic_name1 in topic_name2 or topic_name2 in topic_name1:
                # One contains the other
                shorter = min(len(topic_name1), len(topic_name2))
                longer = max(len(topic_name1), len(topic_name2))
                name_similarity = shorter / longer if longer > 0 else 0.0
            
            # Calculate cosine similarity from embeddings
            emb2 = topic_embeddings[topic_id2]
            embedding_similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            # Use the higher of name similarity or embedding similarity
            # Also connect if either is above threshold
            should_connect = False
            final_similarity = max(name_similarity, embedding_similarity)
            
            if name_similarity >= 0.7:  # Strong name match
                should_connect = True
            elif embedding_similarity >= similarity_threshold:  # Good embedding match
                should_connect = True
            
            if should_connect and not G.has_edge(topic_id1, topic_id2):
                G.add_edge(topic_id1, topic_id2, type="similar_topic", weight=float(final_similarity))
                similarity_edges.append({
                    "source": topic_id1,
                    "target": topic_id2,
                    "type": "similar_topic",
                    "weight": float(final_similarity),
                    "name_similarity": float(name_similarity),
                    "embedding_similarity": float(embedding_similarity)
                })
    
    print(f"  ✅ Found {len(similarity_edges)} topic relationships")
    if len(similarity_edges) > 0:
        print(f"     Average similarity: {np.mean([e['weight'] for e in similarity_edges]):.3f}")
    
    # Convert NetworkX graph to JSON format
    nodes = []
    edges = []
    
    # Add nodes
    for node_id, node_data in G.nodes(data=True):
        node_dict = {"id": node_id}
        node_dict.update({k: v for k, v in node_data.items() if k != 'node_type'})
        nodes.append(node_dict)
    
    # Add edges
    for source, target, edge_data in G.edges(data=True):
        edges.append({
            "source": source,
            "target": target,
            "type": edge_data.get("type", "related"),
            "weight": edge_data.get("weight", 1.0)
        })
    
    graph = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "papers_count": len(papers),
            "topics_count": len(topic_nodes),
            "edges_count": len(edges),
            "paper_topic_edges": sum(1 for e in edges if e["type"] == "contains_topic"),
            "topic_topic_edges": len(similarity_edges)
        }
    }
    
    # Save graph
    graph_path = os.path.join(RESEARCH_GRAPH_DIR, "research_graph.json")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Research graph saved to {graph_path}")
    print(f"   Papers: {len(papers)}, Topics: {len(topic_nodes)}, Edges: {len(edges)}")
    
    return graph


def get_research_graph() -> Dict[str, Any]:
    """Load the research graph from disk."""
    graph_path = os.path.join(RESEARCH_GRAPH_DIR, "research_graph.json")
    
    if not os.path.exists(graph_path):
        return {
            "error": "Research graph not found. Please generate it first.",
            "nodes": [],
            "edges": [],
            "metadata": {
                "papers_count": 0,
                "topics_count": 0,
                "edges_count": 0
            }
        }
    
    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {
            "error": f"Error loading graph: {str(e)}",
            "nodes": [],
            "edges": [],
            "metadata": {
                "papers_count": 0,
                "topics_count": 0,
                "edges_count": 0
            }
        }
