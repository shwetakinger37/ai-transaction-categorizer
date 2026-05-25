
# AI Powered Transaction Categorization Service

## Overview

This project is a backend-only AI-powered transaction categorization system built using Django REST Framework and OpenAI.

The API accepts:
- Transaction details
- Company industry
- Chart of accounts
- Historical categorized transactions

The backend constructs a structured prompt and sends it to an LLM for categorization.

The response includes:
- Suggested category
- Confidence score
- Reasoning

---

## Features

- Django REST API
- OpenAI integration
- LLM abstraction layer
- Structured JSON response
- Service-layer architecture
- Logging and error handling
- Config-driven architecture

---

## Project Structure

```
categorizer/
    views.py
    schemas.py
    urls.py

services/
    context_builder.py
    categorization_service.py
    response_parser.py

llm/
    factory.py
    openai_client.py
    mock_client.py
```

---

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv dja
```

### 2. Activate Environment

```bash
dja\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create `.env` file:

```env
LLM_PROVIDER=openai
LLM_API_KEY=your_openai_key
LLM_MODEL=gpt-4o-mini
DEBUG=True
```

### 5. Run Migrations

```bash
python manage.py migrate
```

### 6. Run Server

```bash
python manage.py runserver
```

---

## API Endpoint

### POST

```
/api/v1/categorize/
```

---

## Example Request

```json
{
  "transaction": {
    "description": "AWS monthly hosting bill",
    "payee": "Amazon Web Services",
    "amount": 250,
    "currency": "USD",
    "date": "2024-05-01"
  },
  "company_id": "company_001",
  "industry": "Software",
  "chart_of_accounts": [
    "Cloud Hosting",
    "Travel Expense",
    "Office Equipment"
  ],
  "historical_transactions": []
}
```

---

## Example Response

```json
{
  "category": "Cloud Hosting",
  "confidence": 0.91,
  "reason": "The transaction is related to cloud infrastructure services."
}
```
