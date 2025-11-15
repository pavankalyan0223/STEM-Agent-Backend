"""
Research Graph Generator
Builds an interactive knowledge graph from research papers using:
- Grobid for PDF extraction (clean text + formulas)
- KeyBERT for keyword extraction using embeddings
- Sentence-BERT (SentenceTransformer) for sentence embeddings
- Ollama/Mistral for sentence comparison to check if sentences describe same concept
- NetworkX for graph structure
"""

import os
import json
import numpy as np
import re
import requests
from typing import Dict, List, Any, Set, Tuple, Optional
from PyPDF2 import PdfReader  # Fallback if Grobid unavailable
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import config for Ollama
try:
    from config import OLLAMA_URL, MODEL_NAME, HTTP_TIMEOUT
except ImportError:
    OLLAMA_URL = "http://localhost:11434/api/chat"
    MODEL_NAME = "mistral:7b-instruct"
    HTTP_TIMEOUT = 60.0

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

try:
    from grobid_client.grobid_client import GrobidClient
    GROBID_AVAILABLE = True
except ImportError:
    GROBID_AVAILABLE = False
    GrobidClient = None

# Alternative: Use Grobid REST API directly
GROBID_REST_AVAILABLE = True  # We'll use requests directly

RESEARCH_GRAPH_DIR = "data/research_graphs"
os.makedirs(RESEARCH_GRAPH_DIR, exist_ok=True)

# Grobid configuration (default: localhost:8070)
GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")

# Initialize models
kw_model = None
embedder = None
grobid_client = None

if KEYBERT_AVAILABLE and SENTENCE_TRANSFORMER_AVAILABLE:
    print("Initializing models...")
    try:
        kw_model = KeyBERT()
        # Use Sentence-BERT model (all-mpnet-base-v2 is a strong Sentence-BERT model)
        embedder = SentenceTransformer("all-mpnet-base-v2")
        print("Models loaded successfully (KeyBERT + Sentence-BERT)")
    except Exception as e:
        print(f"Error loading models: {e}")
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
    print(f"Missing packages: {', '.join(missing)}. Install with: pip install {' '.join(missing)}")

# Initialize Grobid client if available
grobid_client = None
if GROBID_AVAILABLE:
    try:
        grobid_client = GrobidClient(config_path=None, grobid_server=GROBID_URL)
        print(f"Grobid client initialized (server: {GROBID_URL})")
    except Exception as e:
        print(f"Grobid client initialization failed: {e}")
        print("   Will try REST API directly or fall back to PyPDF2")
        grobid_client = None
else:
    print("Grobid client library not available. Will try REST API directly or fall back to PyPDF2")


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, List[str]]:
    """
    Extract text and formulas from a PDF file using Grobid (if available) or PyPDF2 (fallback).
    
    Returns:
        Tuple of (text, formulas_list)
    """
    formulas = []
    
    # Try Grobid REST API first if available
    try:
        # Check if Grobid server is accessible
        grobid_api_url = f"{GROBID_URL}/api/processFulltextDocument"
        
        # Try to use Grobid REST API directly
        with open(pdf_path, 'rb') as pdf_file:
            files = {'input': pdf_file}
            data = {
                'generateIDs': '1',
                'consolidateCitations': '1',
                'teiCoordinates': ['s', 'head', 'note'],
                'segmentSentences': '1'
            }
            
            response = requests.post(grobid_api_url, files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)
                
                # Extract text content
                text_parts = []
                # Namespace for TEI
                ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
                
                # Extract text from body
                for body in root.findall('.//tei:body', ns):
                    for p in body.findall('.//tei:p', ns):
                        para_text = ''.join(p.itertext())
                        if para_text.strip():
                            text_parts.append(para_text.strip())
                
                text = '\n'.join(text_parts)
                
                # Extract formulas (math elements)
                for formula in root.findall('.//tei:formula', ns):
                    formula_text = ''.join(formula.itertext())
                    if formula_text.strip():
                        formulas.append(formula_text.strip())
                
                # Also check for MathML elements
                for math in root.findall('.//{http://www.w3.org/1998/Math/MathML}math'):
                    math_text = ''.join(math.itertext())
                    if math_text.strip():
                        formulas.append(math_text.strip())
                
                if text.strip():
                    print(f"  Extracted {len(formulas)} formulas using Grobid")
                    return text, formulas
    except requests.exceptions.RequestException as e:
        # Grobid server not available or connection failed
        pass
    except Exception as e:
        print(f"  Grobid REST API extraction failed: {e}, falling back to PyPDF2")
    
    # Fallback to PyPDF2
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        # Clean up common PDF extraction artifacts
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces and tabs
        text = re.sub(r'\n\s*\n', '\n', text)  # Remove excessive newlines
        # Fix common PDF extraction issues: broken words across lines
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix hyphenated words split across lines
        return text, formulas
    except Exception as e:
        print(f"  PyPDF2 extraction failed: {e}")
        return "", []


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
        print(f"  KeyBERT error: {e}")
        return []


def check_sentences_same_concept(sentence1: str, sentence2: str) -> Tuple[bool, str]:
    """
    Use Ollama/Mistral to check if two sentences describe the same scientific concept.
    
    Returns:
        Tuple of (is_same_concept: bool, explanation: str)
    """
    prompt = f"""Do these two sentences describe the same scientific concept? Answer yes/no and explain briefly.

Sentence 1: {sentence1}

Sentence 2: {sentence2}

Answer format: Start with "yes" or "no", then provide a brief explanation."""
    
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        answer_text = result.get("message", {}).get("content", "").strip().lower()
        
        # Parse response
        is_same = answer_text.startswith("yes")
        explanation = result.get("message", {}).get("content", "").strip()
        
        return is_same, explanation
    except Exception as e:
        print(f"  Ollama sentence comparison failed: {e}")
        # Fallback: return False if API call fails
        return False, f"Comparison failed: {str(e)}"


def extract_context_lines(keyword: str, text: str, max_lines: int = 5) -> List[str]:
    """
    Extract context lines (sentences) where a keyword appears in the text.
    Returns up to max_lines sentences containing the keyword.
    """
    keyword_clean = keyword.lower().strip()
    text_lower = text.lower()
    
    # Split text into sentences (simple approach)
    sentences = re.split(r'[.!?]+', text)
    
    context_lines = []
    seen_lines = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue
        
        # Check if keyword appears in this sentence
        words = keyword_clean.split()
        if len(words) == 1:
            pattern = r'\b' + re.escape(keyword_clean) + r'\b'
            if re.search(pattern, sentence.lower()):
                sentence_clean = ' '.join(sentence.split())
                if sentence_clean not in seen_lines and len(sentence_clean) > 20:
                    seen_lines.add(sentence_clean)
                    context_lines.append(sentence_clean)
        else:
            # Multi-word phrase
            phrase_pattern = r'\b' + r'\s+'.join([re.escape(w) for w in words]) + r'\b'
            if re.search(phrase_pattern, sentence.lower()):
                sentence_clean = ' '.join(sentence.split())
                if sentence_clean not in seen_lines and len(sentence_clean) > 20:
                    seen_lines.add(sentence_clean)
                    context_lines.append(sentence_clean)
        
        if len(context_lines) >= max_lines:
            break
    
    return context_lines


def extract_topics_from_paper(text: str, paper_title: str, paper_id: str, paper_filename: str) -> List[Dict[str, Any]]:
    """
    Extract topics/keywords from paper using KeyBERT ONLY.
    Processes entire paper text and sections using KeyBERT's extract_keywords method.
    Validates that all keywords actually appear in the source text.
    Also extracts context lines where keywords appear.
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
                # Extract context lines for this keyword
                context_lines = extract_context_lines(keyword_clean, section, max_lines=3)
                
                seen_topics.add(keyword_lower)
                topics.append({
                    "name": keyword_clean,
                    "score": float(score),
                    "description": f"Extracted from section {i+1}",
                    "context_lines": context_lines,
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "paper_filename": paper_filename
                })
    
    # Add top keywords from full text (already validated)
    for keyword, score in keywords_with_scores[:15]:
        keyword_clean = clean_keyword(keyword)
        keyword_lower = keyword_clean.lower().strip()
        
        if keyword_lower not in seen_topics:
            # Extract context lines for this keyword from full text
            context_lines = extract_context_lines(keyword_clean, text, max_lines=5)
            
            seen_topics.add(keyword_lower)
            topics.append({
                "name": keyword_clean,
                "score": float(score),
                "description": f"Main keyword from {paper_title}",
                "context_lines": context_lines,
                "paper_id": paper_id,
                "paper_title": paper_title,
                "paper_filename": paper_filename
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


def build_research_graph(pdf_dir: str = "data/", graph_filename: Optional[str] = None) -> Dict[str, Any]:
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
    
    print(f"Processing {len(pdf_files)} research papers...")
    
    # Process each paper
    papers = []
    all_topics = []
    paper_texts = {}  # Store full text for each paper for context analysis
    
    for idx, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"\n[{idx+1}/{len(pdf_files)}] Processing: {pdf_file}")
        
        try:
            text, formulas = extract_text_from_pdf(pdf_path)
            if not text.strip():
                print(f"  Could not extract text from {pdf_file}")
                continue
            
            # Extract metadata
            metadata = extract_paper_metadata(text, pdf_file)
            paper_id = f"paper_{idx}"
            
            # Store paper text and formulas for later analysis
            paper_texts[paper_id] = text
            
            # Extract topics using KeyBERT with context lines
            print(f"  Extracting topics with KeyBERT...")
            topics = extract_topics_from_paper(text, metadata['title'], paper_id, pdf_file)
            
            # Get keywords for paper
            keywords_with_scores = extract_keywords_keybert(text, top_n=15)
            keywords = [kw[0] for kw in keywords_with_scores]
            
            # Create paper node with formulas
            paper_node = {
                "id": paper_id,
                "type": "paper",
                "label": metadata['title'],
                "title": metadata['title'],
                "abstract": metadata['abstract'],
                "filename": pdf_file,
                "topics_count": len(topics),
                "keywords": keywords,
                "formulas": formulas,  # Store extracted formulas
                "formulas_count": len(formulas)
            }
            
            papers.append(paper_node)
            all_topics.append(topics)
            
            print(f"  Found {len(topics)} topics")
            
        except Exception as e:
            print(f"  Error processing {pdf_file}: {e}")
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
    topic_contexts = {}  # Store context lines for each topic per paper
    
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
                
                # Add context lines for this paper
                if topic_id not in topic_contexts:
                    topic_contexts[topic_id] = []
                topic_contexts[topic_id].append({
                    "paper_id": paper_id,
                    "paper_title": topic.get('paper_title', ''),
                    "paper_filename": topic.get('paper_filename', ''),
                    "context_lines": topic.get('context_lines', [])
                })
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
                
                # Store context lines for this topic
                topic_contexts[topic_id] = [{
                    "paper_id": paper_id,
                    "paper_title": topic.get('paper_title', ''),
                    "paper_filename": topic.get('paper_filename', ''),
                    "context_lines": topic.get('context_lines', [])
                }]
                
                # Create embedding for this topic (only once per unique topic)
                if embedder:
                    topic_embedding = embedder.encode(topic_name)
                    topic_embeddings[topic_id] = topic_embedding
            
            # Create paper->topic edge
            if not G.has_edge(paper_id, topic_id):
                G.add_edge(paper_id, topic_id, type="contains_topic", weight=1.0)
    
    # Add context information to topic nodes
    for topic_id, contexts in topic_contexts.items():
        if topic_id in topic_nodes:
            topic_nodes[topic_id]["contexts"] = contexts
            # Update node in graph
            G.nodes[topic_id]["contexts"] = contexts
    
    # Find similar topics using embeddings and name matching
    print(f"\nFinding topic similarities using embeddings...")
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
    
    print(f"  Found {len(similarity_edges)} topic relationships")
    if len(similarity_edges) > 0:
        print(f"     Average similarity: {np.mean([e['weight'] for e in similarity_edges]):.3f}")
    
    # Analyze paper relationships based on keyword contexts and sentence comparison
    print(f"\nAnalyzing paper relationships using embeddings and Ollama sentence comparison...")
    paper_relationships = []
    paper_embeddings = {}
    paper_sentences = {}  # Store sentences for each paper
    
    # Create embeddings for paper contexts and extract sentences
    for paper_id in [p['id'] for p in papers]:
        paper_contexts = []
        sentences = []
        
        for topic_id, contexts in topic_contexts.items():
            for ctx in contexts:
                if ctx['paper_id'] == paper_id:
                    paper_contexts.extend(ctx['context_lines'])
                    sentences.extend(ctx['context_lines'])
        
        # Store sentences for this paper
        paper_sentences[paper_id] = sentences[:20]  # Store up to 20 sentences
        
        # Combine contexts and create embedding using Sentence-BERT
        if paper_contexts and embedder:
            combined_context = ' '.join(paper_contexts[:10])  # Use up to 10 context lines
            if len(combined_context) > 50:  # Only if we have enough context
                paper_embeddings[paper_id] = embedder.encode(combined_context)
    
    # Compare papers using context embeddings and Ollama sentence comparison
    paper_ids = list(paper_embeddings.keys())
    relationship_threshold = 0.5  # Threshold for paper similarity
    
    print(f"  Comparing {len(paper_ids)} papers...")
    
    for i, paper_id1 in enumerate(paper_ids):
        emb1 = paper_embeddings[paper_id1]
        sentences1 = paper_sentences.get(paper_id1, [])
        
        for paper_id2 in paper_ids[i+1:]:
            emb2 = paper_embeddings[paper_id2]
            sentences2 = paper_sentences.get(paper_id2, [])
            
            # Calculate cosine similarity using Sentence-BERT embeddings
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            # Also check shared keywords
            paper1_topics = set()
            paper2_topics = set()
            
            for topic_id, contexts in topic_contexts.items():
                paper_ids_in_topic = [ctx['paper_id'] for ctx in contexts]
                if paper_id1 in paper_ids_in_topic:
                    paper1_topics.add(topic_id)
                if paper_id2 in paper_ids_in_topic:
                    paper2_topics.add(topic_id)
            
            shared_topics = paper1_topics.intersection(paper2_topics)
            shared_topics_count = len(shared_topics)
            
            # Use Ollama to check if sentences describe the same concept
            same_concept_count = 0
            total_comparisons = 0
            max_comparisons = min(5, len(sentences1), len(sentences2))  # Compare up to 5 sentence pairs
            
            if sentences1 and sentences2 and max_comparisons > 0:
                print(f"    Comparing sentences between {paper_id1} and {paper_id2}...")
                for idx in range(max_comparisons):
                    try:
                        is_same, explanation = check_sentences_same_concept(
                            sentences1[idx], 
                            sentences2[idx]
                        )
                        total_comparisons += 1
                        if is_same:
                            same_concept_count += 1
                    except Exception as e:
                        print(f"      Sentence comparison error: {e}")
                        continue
            
            # Calculate same concept ratio
            same_concept_ratio = same_concept_count / total_comparisons if total_comparisons > 0 else 0.0
            
            # Determine if papers are related
            # Papers are related if they share keywords AND have similar contexts OR same concepts
            is_related = False
            relationship_strength = 0.0
            
            if shared_topics_count > 0 or same_concept_ratio > 0.3:
                # Calculate relationship strength based on shared topics, context similarity, and same concepts
                topic_weight = min(shared_topics_count / 5.0, 1.0)  # Normalize by max 5 shared topics
                context_weight = max(0, similarity)  # Use similarity if positive
                concept_weight = same_concept_ratio  # Ratio of sentences describing same concept
                
                # Weighted combination: topics (40%), embeddings (30%), same concepts (30%)
                relationship_strength = (topic_weight * 0.4 + context_weight * 0.3 + concept_weight * 0.3)
                
                # Papers are related if they share at least 2 topics OR have high context similarity OR high same concept ratio
                if shared_topics_count >= 2 or similarity >= relationship_threshold or same_concept_ratio >= 0.4:
                    is_related = True
            
            if is_related and not G.has_edge(paper_id1, paper_id2):
                G.add_edge(paper_id1, paper_id2, 
                          type="related_paper", 
                          weight=float(relationship_strength),
                          shared_topics_count=shared_topics_count,
                          context_similarity=float(similarity),
                          same_concept_ratio=float(same_concept_ratio))
                paper_relationships.append({
                    "source": paper_id1,
                    "target": paper_id2,
                    "type": "related_paper",
                    "weight": float(relationship_strength),
                    "shared_topics_count": shared_topics_count,
                    "context_similarity": float(similarity),
                    "same_concept_ratio": float(same_concept_ratio)
                })
    
    print(f"  Found {len(paper_relationships)} paper relationships")
    if len(paper_relationships) > 0:
        avg_strength = np.mean([r['weight'] for r in paper_relationships])
        print(f"     Average relationship strength: {avg_strength:.3f}")
    
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
        edge_dict = {
            "source": source,
            "target": target,
            "type": edge_data.get("type", "related"),
            "weight": edge_data.get("weight", 1.0)
        }
        # Add additional metadata for paper-paper relationships
        if edge_data.get("type") == "related_paper":
            edge_dict["shared_topics_count"] = edge_data.get("shared_topics_count", 0)
            edge_dict["context_similarity"] = edge_data.get("context_similarity", 0.0)
            edge_dict["same_concept_ratio"] = edge_data.get("same_concept_ratio", 0.0)
        edges.append(edge_dict)
    
    graph = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "papers_count": len(papers),
            "topics_count": len(topic_nodes),
            "edges_count": len(edges),
            "paper_topic_edges": sum(1 for e in edges if e["type"] == "contains_topic"),
            "topic_topic_edges": len(similarity_edges),
            "paper_paper_edges": len(paper_relationships)
        }
    }
    
    # Save graph with custom filename or default
    if graph_filename:
        # Ensure filename ends with .json
        if not graph_filename.endswith('.json'):
            graph_filename += '.json'
        graph_path = os.path.join(RESEARCH_GRAPH_DIR, graph_filename)
    else:
        # Default filename
        graph_path = os.path.join(RESEARCH_GRAPH_DIR, "research_graph.json")
    
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    
    print(f"\nResearch graph saved to {graph_path}")
    print(f"   Papers: {len(papers)}, Topics: {len(topic_nodes)}, Edges: {len(edges)}")
    
    # Add filename to graph metadata
    graph["metadata"]["filename"] = os.path.basename(graph_path)
    
    return graph


def get_research_graph(graph_filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a research graph from disk.
    
    Args:
        graph_filename: Optional filename of the graph to load. If None, loads default "research_graph.json"
    """
    if graph_filename:
        # Ensure filename ends with .json
        if not graph_filename.endswith('.json'):
            graph_filename += '.json'
        graph_path = os.path.join(RESEARCH_GRAPH_DIR, graph_filename)
    else:
        graph_path = os.path.join(RESEARCH_GRAPH_DIR, "research_graph.json")
    
    if not os.path.exists(graph_path):
        return {
            "error": f"Research graph '{os.path.basename(graph_path)}' not found. Please generate it first.",
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
            graph_data = json.load(f)
            # Ensure filename is in metadata
            if "metadata" in graph_data:
                graph_data["metadata"]["filename"] = os.path.basename(graph_path)
            return graph_data
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


def delete_research_graph(graph_filename: str) -> Dict[str, Any]:
    """
    Delete a research graph file.
    
    Args:
        graph_filename: Filename of the graph to delete
    """
    if not graph_filename:
        return {"error": "Graph filename is required"}
    
    # Ensure filename ends with .json
    if not graph_filename.endswith('.json'):
        graph_filename += '.json'
    
    graph_path = os.path.join(RESEARCH_GRAPH_DIR, graph_filename)
    
    if not os.path.exists(graph_path):
        return {"error": f"Graph '{graph_filename}' not found"}
    
    try:
        os.remove(graph_path)
        return {"success": True, "message": f"Graph '{graph_filename}' deleted successfully"}
    except Exception as e:
        return {"error": f"Error deleting graph: {str(e)}"}
