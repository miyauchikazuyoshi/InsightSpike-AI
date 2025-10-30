---
status: active
category: infra
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
next_step: Introduce unified rate limit + retry middleware for external APIs.
---

# External API Integration Plan for Document Processing

## Overview

This plan outlines the integration of external APIs for document processing capabilities, allowing InsightSpike to focus on its core strength of insight discovery while leveraging best-in-class tools for document handling.

## Goals

1. Support multiple document formats (PDF, DOCX, HTML, etc.) without reinventing parsers
2. Enable scalable batch processing through external services
3. Maintain InsightSpike's lightweight architecture
4. Focus development on unique insight discovery features

## Architecture Design

### Plugin-Based Document Processing

```python
class DocumentProcessorPlugin:
    """Base class for document processor plugins"""
    
    def can_process(self, file_path: str) -> bool:
        """Check if this plugin can handle the file"""
        pass
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from document"""
        pass
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract document metadata"""
        pass

class DocumentProcessorRegistry:
    """Registry for document processors"""
    
    def __init__(self):
        self.processors = {}
        
    def register(self, extensions: List[str], processor: DocumentProcessorPlugin):
        for ext in extensions:
            self.processors[ext] = processor
    
    def process(self, file_path: str) -> ProcessedDocument:
        ext = Path(file_path).suffix.lower()
        if ext in self.processors:
            return self.processors[ext].process(file_path)
        raise ValueError(f"No processor for {ext}")
```

## External Service Options

### 1. PDF Processing

**Option A: LlamaParse (Recommended)**
- Pros: AI-native, handles complex layouts, tables, images
- Cons: Requires API key, costs per page
- Integration:
```python
class LlamaParsePlugin(DocumentProcessorPlugin):
    def __init__(self, api_key: str):
        self.client = LlamaParse(api_key=api_key)
    
    def extract_text(self, file_path: str):
        result = self.client.parse(file_path)
        return result.text
```

**Option B: Unstructured.io**
- Pros: Open source option available, multiple formats
- Cons: Self-hosted version requires resources
- Integration: REST API or Python SDK

**Option C: Apache Tika**
- Pros: Free, comprehensive format support
- Cons: Requires Java, heavier footprint
- Integration: Python wrapper (tika-python)

### 2. OCR for Scanned Documents

**Option A: AWS Textract**
- Pros: Accurate, handles forms and tables
- Cons: AWS dependency, costs
- Use case: Enterprise deployments

**Option B: Google Cloud Vision**
- Pros: High accuracy, multiple languages
- Cons: Google Cloud dependency
- Use case: When already using GCP

**Option C: Tesseract (via pytesseract)**
- Pros: Free, open source
- Cons: Lower accuracy, requires local installation
- Use case: Budget-conscious deployments

### 3. Text Chunking

**LangChain TextSplitters (Recommended)**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

class LangChainChunkingPlugin:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)
```

## Implementation Plan

### Phase 1: Core Integration Framework (Week 1-2)

1. Create plugin architecture
2. Implement registry system
3. Add configuration for external services
4. Update CLI to support `--processor` flag

```bash
# New CLI syntax
spike embed docs/ --processor llamaparse --chunk-size 500
```

### Phase 2: Service Integrations (Week 3-4)

1. Implement LlamaParse plugin
2. Implement LangChain chunking
3. Add Tesseract OCR plugin
4. Create fallback mechanisms

### Phase 3: Advanced Features (Week 5-6)

1. Batch processing with progress tracking
2. Resume capability for interrupted jobs
3. Caching of processed documents
4. Metadata extraction and storage

## Configuration

### Environment Variables
```bash
# External service credentials
LLAMAPARSE_API_KEY=xxx
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx

# Service selection
INSIGHTSPIKE_PDF_PROCESSOR=llamaparse
INSIGHTSPIKE_OCR_SERVICE=tesseract
INSIGHTSPIKE_CHUNKING_SERVICE=langchain
```

### Config File (config.yaml)
```yaml
document_processing:
  pdf_processor: llamaparse
  ocr_service: tesseract
  chunking:
    service: langchain
    chunk_size: 500
    chunk_overlap: 50
  
  # Service-specific settings
  llamaparse:
    api_key: ${LLAMAPARSE_API_KEY}
    parsing_mode: "accurate"
  
  tesseract:
    language: "eng"
    config: "--oem 3 --psm 6"
```

## CLI Enhancement

### New Commands
```bash
# Process documents with external services
spike process <path> --format auto --output processed/

# List available processors
spike processors list

# Test processor on single file
spike processors test <file> --processor llamaparse
```

### Enhanced Embed Command
```bash
spike embed <path> \
  --processor llamaparse \
  --chunk-size 500 \
  --chunk-overlap 50 \
  --extract-metadata \
  --cache-processed
```

## Error Handling

1. **Service Unavailable**: Fall back to basic text extraction
2. **Rate Limiting**: Implement exponential backoff
3. **Large Files**: Stream processing where possible
4. **Format Issues**: Clear error messages with suggestions

## Cost Considerations

| Service | Cost Model | Estimated Monthly Cost |
|---------|-----------|----------------------|
| LlamaParse | $0.10/page | $10-100 for moderate use |
| AWS Textract | $1.50/1000 pages | $15-150 based on volume |
| Unstructured.io | Self-hosted free | Infrastructure costs only |
| Tesseract | Free | $0 |

## Success Metrics

1. Support for 10+ document formats
2. Processing speed: 100+ documents/minute
3. Text extraction accuracy: >95%
4. Zero maintenance of parsing code
5. Easy addition of new processors

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| API service downtime | Multiple service options, fallbacks |
| API cost overruns | Usage monitoring, quotas |
| Privacy concerns | Option for local-only processing |
| Integration complexity | Plugin architecture, clear interfaces |

## Conclusion

By leveraging external APIs for document processing, InsightSpike can:
1. Immediately support numerous formats
2. Avoid maintaining complex parsing code
3. Focus on core insight discovery features
4. Provide flexibility for different deployment scenarios

This approach aligns with InsightSpike's philosophy of being a lightweight, focused tool that excels at insight discovery rather than trying to be a complete document processing platform.