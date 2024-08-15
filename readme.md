
```markdown
# PDF Document Processing and QA Bot

This project provides a Flask-based web service for uploading PDF documents, extracting text using OCR, and setting up a question-answering (QA) system using language models and embeddings. The service processes uploaded PDFs, stores embeddings in a Chroma database, and allows users to query the processed documents.

## Features

- **PDF Upload**: Upload PDF documents through a REST API endpoint.
- **Text Extraction**: Extract text from PDF files using PaddleOCR.
- **Text Splitting**: Split extracted text into manageable chunks.
- **Embedding**: Convert text chunks into embeddings using HuggingFace's model.
- **Database Storage**: Store embeddings in a Chroma vector database.
- **Question Answering**: Query the processed documents using a custom QA chain.

## Prerequisites

- Python 3.7 or higher
- Required Python packages (see `requirements.txt`)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gautamraj8044/PDF-Document-Processing-and-QA-Bot
   ```

2. **Navigate to the project directory:**
   ```bash
   cd repositoryname
   ```

3. **Install the required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Set up Poppler**: Download and install Poppler, and update the `poppler_path` variable in the code to point to your Poppler installation directory.

2. **Configure Model Paths**: Update the `local_llm` variable with the path to your local language model file.

## Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Access the API:**
   - **Upload a PDF**: POST request to `/upload` with a file attachment.
   - **Ask a Question**: POST request to `/ask` with the question in the form data.

## API Endpoints

### Upload PDF

- **Endpoint**: `/upload`
- **Method**: POST
- **Request**: Form-data with a file attachment.
- **Response**: JSON message indicating the status of the upload and processing.

### Ask Question

- **Endpoint**: `/ask`
- **Method**: POST
- **Request**: Form-data with the key `query` containing the question.
- **Response**: JSON with the answer to the question.

## Example Usage

### Upload a PDF

```bash
curl -X POST http://localhost:5000/upload -F "file=@path_to_your_pdf.pdf"
```

### Ask a Question

```bash
curl -X POST http://localhost:5000/ask -F "query=What is the main topic of the document?"
```



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to submit issues or pull requests. Please follow the project's coding style and guidelines.

## Contact

For any questions or issues, please contact [gautamraj8044@gmail.com]