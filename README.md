# Prodigal Payment Analyzer (Streamlit + GROQ)

AI system to analyze debt-collection call transcripts, extract payment attempts, validate card details via Prodigal’s API, and display color‑coded outcomes.

- Student (SAP) ID: 70022200506  
- Live Demo: <your Streamlit URL here>  
- GitHub: https://github.com/AakashPathak07/prodigal-payment-analyzer

## Features

- Task 1: Call Analysis
  - Payment attempted (boolean)
  - Customer intent (boolean, rule‑backed)
  - Sentiment classification with explanation
  - Agent performance notes
  - Timestamped key events (disclosures, negotiations, attempts, frustration)

- Task 2: Payment Validation
  - LLM extraction of cardholder name, PAN, expiry, CVV, amount
  - Deterministic validity + failure reasons mapping
  - Handles multiple attempts by extracting the latest complete attempt
  - Calls Prodigal Payment Validation API and shows responses
  - Color‑coded status with debug toggles

## Quick Start

