# Testing Guide - Complete System (v4.0 Final)

**Status**: Production Ready - All Phases Complete
**System**: Multimodal RAG for Test Case Generation
**Last Updated**: January 2, 2026

---

## Pre-Testing Checklist

Before testing, ensure you have:
- [ ] Python 3.10+ installed
- [ ] OpenAI API key ready
- [ ] Virtual environment activated
- [ ] Dependencies installed

---

## Step 1: Environment Setup

### 1.1 Navigate to Project

```bash
cd D:\multimodal-rag-system
```

### 1.2 Create Virtual Environment

```bash
python -m venv venv
```

### 1.3 Activate Virtual Environment

**Windows (PowerShell/CMD)**:
```bash
venv\Scripts\activate
```

**Linux/Mac**:
```bash
source venv/bin/activate
```

**Verify activation** - you should see `(venv)` in your terminal.

### 1.4 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected output**:
```
Successfully installed openai-1.12.0 faiss-cpu-1.7.4 ...
```

This will take 2-3 minutes.

### 1.5 Create .env File

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

### 1.6 Add API Key

Edit `.env` file and replace `your_openai_api_key_here` with your actual key:

```bash
OPENAI_API_KEY=sk-proj-...your-actual-key...
```

**Save the file.**

---

## Step 2: Verify Installation

### 2.1 Check Python Version

```bash
python --version
```

**Expected**: Python 3.10.0 or higher

### 2.2 Test Imports

```bash
python -c "import openai; import faiss; import pdfplumber; print('✓ All packages working')"
```

**Expected**: `✓ All packages working`

**If error**: Make sure venv is activated

### 2.3 Check Project Structure

```bash
# Windows
dir /S /B src

# Linux/Mac
find src -type f -name "*.py" | head -20
```

**Expected**: Should list Python files in src/

---

## Step 3: Test Ingestion

### 3.1 Run Ingestion

```bash
python src/main.py --ingest data/sample_dataset/user-signup/
```

### 3.2 Expected Output

```
Initializing RAG system...

============================================================
📥 INGESTION MODE
============================================================

📂 Ingesting 3 files from data/sample_dataset/user-signup
============================================================
Processing files: 100%|██████████| 3/3

✓ PRD_Signup.md: 12 chunks created
✓ API_Spec.yaml: 8 chunks created
✓ Error_Codes.txt: 15 chunks created

============================================================
📊 Ingestion Summary
============================================================
Files processed:      3
Documents created:    3
Chunks created:       35
Embeddings generated: 35
============================================================

✓ Vector store saved to data/storage/vector_db
  - Total vectors: 35
  - Dimension: 1536
```

### 3.3 Verify Files Created

```bash
# Windows
dir data\storage\vector_db

# Linux/Mac
ls -lh data/storage/vector_db/
```

**Expected files**:
- `faiss.index` (vector index)
- `metadata.pkl` (chunk metadata)

### 3.4 Test Statistics

```bash
python src/main.py --stats
```

**Expected output**:
```
============================================================
📊 VECTOR STORE STATISTICS
============================================================
total_vectors: 35
dimension: 1536
total_chunks: 35
unique_sources: 3
```

---

## Step 4: Test Querying

### 4.1 Test Query 1: Basic Query

```bash
python src/main.py --query "Create use cases for user signup"
```

### 4.2 Expected Output

```
============================================================
🔍 QUERY MODE
============================================================
Query: Create use cases for user signup

============================================================
📊 RESULTS
============================================================

Retrieved chunks: 5
Average score: 0.7845

Generated use cases: 3

============================================================
📝 USE CASES (Summary)
============================================================

1. Successful signup with valid email and password
   Goal: Verify user can create account with valid credentials
   Steps: 4
   Negative cases: 1
   Boundary cases: 1

2. Reject duplicate email signup
   Goal: Verify system rejects duplicate email addresses
   Steps: 3
   Negative cases: 0
   Boundary cases: 0

3. Password validation test
   Goal: Verify password meets security requirements
   Steps: 4
   Negative cases: 2
   Boundary cases: 1
```

### 4.3 Test Query 2: Specific Query

```bash
python src/main.py --query "Generate negative test cases for email validation"
```

**Expected**: Should return test cases focused on negative email scenarios

### 4.4 Test Query 3: Debug Mode

```bash
python src/main.py --query "Create boundary test cases" --debug
```

**Expected**: Additional debug output showing:
- Query embedding generation
- Retrieved chunks with scores
- Full JSON output at the end

### 4.5 Test Query 4: Save to File

```bash
python src/main.py --query "Create test cases for error handling" --output test_output.json
```

**Expected**:
- Normal output displayed
- File `test_output.json` created
- Message: `✓ Output saved to test_output.json`

**Verify file**:
```bash
# Windows
type test_output.json

# Linux/Mac
cat test_output.json
```

Should contain valid JSON with use cases.

---

## Step 5: Test Helper Scripts

### 5.1 Test ingest.py

```bash
python scripts/ingest.py data/sample_dataset/user-signup/
```

**Expected**: Same as main ingestion (but simpler interface)

### 5.2 Test query.py

```bash
python scripts/query.py "Create use cases for signup"
```

**Expected**: Should display use cases

---

## Step 6: Validate Output Structure

### 6.1 Check JSON Structure

Open `test_output.json` and verify it contains:

```json
{
  "use_cases": [
    {
      "title": "...",
      "goal": "...",
      "preconditions": [...],
      "test_data": {...},
      "steps": [...],
      "expected_results": [...],
      "negative_cases": [...],
      "boundary_cases": [...]
    }
  ],
  "assumptions": [...],
  "missing_information": [...],
  "confidence_score": 0.XX
}
```

### 6.2 Validate Required Fields

Each use case MUST have:
- ✅ title (string)
- ✅ goal (string)
- ✅ preconditions (list)
- ✅ steps (list, non-empty)
- ✅ expected_results (list, non-empty)

---

## Step 7: Error Testing

### 7.1 Test Without Ingestion

Delete vector store:
```bash
# Windows
rmdir /S /Q data\storage\vector_db

# Linux/Mac
rm -rf data/storage/vector_db
```

Try to query:
```bash
python src/main.py --query "test"
```

**Expected error**:
```
❌ No documents in vector store. Please ingest documents first.
   Run: python src/main.py --ingest data/sample_dataset/user-signup/
```

**Re-ingest** to continue testing:
```bash
python src/main.py --ingest data/sample_dataset/user-signup/
```

### 7.2 Test Invalid API Key

Edit `.env` and set invalid key:
```
OPENAI_API_KEY=invalid_key
```

Try to ingest:
```bash
python src/main.py --ingest data/sample_dataset/user-signup/
```

**Expected**: Error about invalid API key

**Fix**: Restore correct API key in `.env`

### 7.3 Test Invalid Folder Path

```bash
python src/main.py --ingest nonexistent/folder/
```

**Expected error**:
```
FileNotFoundError: Folder not found: nonexistent/folder/
```

---

## Step 8: Functional Tests

### Test Case 1: Context Grounding

**Query**: "Create use cases for features not in documentation"

**Expected**: System should either:
- Say insufficient context
- Only generate from existing docs
- NOT invent features

### Test Case 2: Multiple Queries

Run 3-5 different queries and verify:
- ✅ Different use cases generated each time
- ✅ Use cases relevant to query
- ✅ Steps are specific and actionable
- ✅ No hallucinated information

### Test Case 3: Edge Cases

**Empty folder**:
```bash
mkdir empty_folder
python src/main.py --ingest empty_folder/
```

**Expected**: "No supported files found"

---

## Step 9: Performance Check

### 9.1 Measure Ingestion Time

```bash
# Windows (PowerShell)
Measure-Command {python src/main.py --ingest data/sample_dataset/user-signup/}

# Linux/Mac
time python src/main.py --ingest data/sample_dataset/user-signup/
```

**Expected**: < 10 seconds for 3 files

### 9.2 Measure Query Time

```bash
time python src/main.py --query "Create use cases"
```

**Expected**: < 15 seconds end-to-end

---

## Step 10: Code Quality Checks (Optional)

### 10.1 Run Black (Code Formatting)

```bash
black src/ --check
```

### 10.2 Run Flake8 (Linting)

```bash
flake8 src/ --max-line-length=100
```

### 10.3 Run MyPy (Type Checking)

```bash
mypy src/ --ignore-missing-imports
```

---

## Test Results Checklist

### ✅ Must Pass:
- [ ] Environment setup successful
- [ ] Dependencies installed
- [ ] Ingestion completes without errors
- [ ] Vector store files created
- [ ] Basic query returns use cases
- [ ] JSON output structure is valid
- [ ] Use cases have all required fields
- [ ] Debug mode shows retrieved chunks
- [ ] Output can be saved to file
- [ ] Helper scripts work

### ✅ Should Pass:
- [ ] Multiple queries return different results
- [ ] Use cases are grounded in context
- [ ] No obvious hallucinations
- [ ] Error handling works (invalid folder, no API key)
- [ ] Performance is acceptable (<15s per query)

### 🔧 Known Issues (Expected):
- Image files not supported yet (Phase 2)
- DOCX files not supported yet (Phase 2)
- No BM25 search yet (Phase 2)
- No hallucination detection (Phase 3)
- No unit tests (Phase 3)

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution**:
```bash
# Make sure you're in project root
cd D:\multimodal-rag-system

# Make sure venv is activated
venv\Scripts\activate
```

### Issue: "OPENAI_API_KEY not found"

**Solution**:
```bash
# Check .env file exists
dir .env  # Windows
ls .env   # Linux/Mac

# Check content
type .env  # Windows
cat .env   # Linux/Mac

# Make sure key is set correctly
```

### Issue: "faiss module not found"

**Solution**:
```bash
pip install faiss-cpu==1.7.4
```

### Issue: "pdfplumber module not found"

**Solution**:
```bash
pip install pdfplumber==0.10.3
```

### Issue: Empty use cases returned

**Possible causes**:
1. Low similarity scores - try --debug to see scores
2. No documents ingested - run --stats to check
3. Query too vague - be more specific

---

## Success Criteria

### ✅ Phase 1 Testing is successful if:

1. **Ingestion works**: Files are parsed, chunked, embedded, and stored
2. **Retrieval works**: Queries return relevant chunks with good scores
3. **Generation works**: LLM produces structured JSON use cases
4. **Output is valid**: JSON structure matches schema
5. **Use cases are grounded**: No obvious hallucinations
6. **Performance is acceptable**: Queries complete in <15s
7. **Error handling works**: Invalid inputs produce clear errors

---

## Next Steps After Testing

### If All Tests Pass ✅:
1. Commit to Git
2. Start Phase 2 (Image OCR, BM25, Hybrid retrieval)
3. Create sample demo video

### If Some Tests Fail ⚠️:
1. Document the failures
2. Check error messages
3. Fix bugs
4. Re-test
5. Iterate until passing

### If Major Issues ❌:
1. Review PHASE1_PROGRESS_REPORT.md
2. Check implementation against TECHNICAL_SPECIFICATION.md
3. Debug with --debug flag
4. Check logs
5. Ask for help if needed

---

## Phase 4: Web Interface Testing (NEW)

### Test 1: Web Server Startup

**Objective**: Verify web server starts successfully

```bash
python run_web.py
```

**Expected Output**:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Validation**:
- [ ] Server starts without errors
- [ ] No import errors
- [ ] Port 8000 is accessible

### Test 2: Web UI Access

**Objective**: Verify web interface loads

**Steps**:
1. Open browser
2. Navigate to http://localhost:8000/app

**Expected Result**:
- [ ] Page loads successfully
- [ ] Title shows "Multimodal RAG System"
- [ ] Upload section visible
- [ ] Query section visible
- [ ] Statistics button visible
- [ ] Tailwind CSS styling applied

### Test 3: API Documentation Access

**Objective**: Verify Swagger/OpenAPI docs accessible

**Steps**:
1. Navigate to http://localhost:8000/docs

**Expected Result**:
- [ ] Swagger UI loads
- [ ] Endpoints visible: /upload, /query, /stats, /health
- [ ] Can expand endpoint details
- [ ] Schemas visible

### Test 4: File Upload via Web UI

**Objective**: Test document upload through drag-and-drop interface

**Steps**:
1. Navigate to http://localhost:8000/app
2. Click "Choose files" or drag PDF file
3. Click "Upload" button

**Expected Result**:
- [ ] File preview appears
- [ ] Upload progress shows
- [ ] Success message displayed
- [ ] Document appears in list
- [ ] File type icon shown

**Sample Test File**: `data/sample_dataset/user-signup/PRD_Signup.md`

### Test 5: Image Upload with Vision API

**Objective**: Test image upload and Vision API processing

**Steps**:
1. Navigate to http://localhost:8000/app
2. Drag image file (PNG/JPG) to upload zone
3. Verify image thumbnail preview appears
4. Click "Upload"
5. Wait for processing

**Expected Result**:
- [ ] Image preview shown
- [ ] Upload completes successfully
- [ ] Success message: "Image processed with OCR/Vision API"
- [ ] Vector count increases (check with stats button)
- [ ] Image appears in documents list with image icon

**Configuration Check**:
Verify `.env` has:
```
USE_VISION_API=true
VISION_MODEL=gpt-4o-mini
```

### Test 6: Query via Web Interface

**Objective**: Test query processing through web UI

**Steps**:
1. Upload sample documents first (if not already done)
2. Type query in query box: "Create use cases for user signup"
3. Click "Generate Test Cases"

**Expected Result**:
- [ ] Query processes successfully
- [ ] JSON output displayed in result area
- [ ] Output contains use_cases array
- [ ] Each use case has required fields
- [ ] Copy button works
- [ ] No errors in console

### Test 7: Statistics Dashboard

**Objective**: Verify statistics endpoint works

**Steps**:
1. Click "View Statistics" button in web UI

**Expected Result**:
- [ ] Modal/alert shows statistics
- [ ] Total vectors displayed
- [ ] Dimension shown (1536)
- [ ] BM25 indexed status shown
- [ ] Unique sources count displayed

### Test 8: API Endpoint Testing

**Objective**: Test REST API endpoints directly

**Method 1: Using Swagger UI**
1. Navigate to http://localhost:8000/docs
2. Expand `/upload` endpoint
3. Click "Try it out"
4. Upload test file
5. Click "Execute"

**Expected Result**:
- [ ] 200 status code
- [ ] Response shows chunks_added, total_vectors
- [ ] No errors

**Method 2: Using curl**

```bash
# Health check
curl http://localhost:8000/health

# Stats
curl http://localhost:8000/stats

# Upload (Windows PowerShell)
curl -X POST "http://localhost:8000/upload" -F "file=@data/sample_dataset/user-signup/PRD_Signup.md"

# Query
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d "{\"query\": \"Create test cases\"}"
```

### Test 9: Image Query Relevance

**Objective**: Verify image content is retrievable

**Steps**:
1. Upload an image with specific content (e.g., flight booking screenshot)
2. Wait for processing
3. Query with specific terms from the image
4. Example query: "Create test cases for flight filter"

**Expected Result**:
- [ ] Query retrieves content from uploaded image
- [ ] Generated test cases reference image content
- [ ] Confidence scores reasonable (>0.3)

**Note**: Generic queries may match other documents better. Use specific terms from the image for best results.

### Test 10: Error Handling

**Objective**: Verify graceful error handling

**Tests**:

**a) Invalid File Type**:
1. Try uploading .exe file
**Expected**: Error message, upload rejected

**b) Empty Query**:
1. Submit query with empty text
**Expected**: Error message or prompt to enter query

**c) No Documents**:
1. Clear vector store
2. Try querying
**Expected**: Message about no documents

**d) Server Restart**:
1. Stop server (Ctrl+C)
2. Restart `python run_web.py`
**Expected**: Clean restart, no data loss

### Web Interface Test Results Summary

```
Total Web Tests: 10
☑ Server Startup
☑ Web UI Access
☑ API Docs Access
☑ File Upload
☑ Image Upload with Vision
☑ Query Interface
☑ Statistics Dashboard
☑ API Endpoints
☑ Image Query Relevance
☑ Error Handling

Overall Web Interface Status: [PASS/FAIL]
```

---

## Test Log Template

```markdown
# Test Log - YYYY-MM-DD

## Environment
- Python version:
- OS:
- Date:

## Test Results

### Ingestion Test
- Status: PASS/FAIL
- Files processed:
- Chunks created:
- Time taken:
- Notes:

### Query Test 1
- Query:
- Status: PASS/FAIL
- Use cases generated:
- Quality: GOOD/FAIR/POOR
- Time taken:
- Notes:

### Query Test 2
- Query:
- Status: PASS/FAIL
- Use cases generated:
- Quality: GOOD/FAIR/POOR
- Time taken:
- Notes:

### Overall Assessment
- Total tests:
- Passed:
- Failed:
- Ready for Phase 2: YES/NO
- Comments:
```

---

**Testing Guide Complete**
**Next Action**: Run through all test steps
**Document**: Log results in test log
