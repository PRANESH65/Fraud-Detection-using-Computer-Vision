from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import time

# Initialize the FastAPI app
app = FastAPI()

# MongoDB connection (local or MongoDB Atlas)
try:
    client = MongoClient("mongodb://localhost:27017/")  # For local MongoDB

    # Connect to the 'fraud_detection' database and 'transactions' collection
    db = client['fraud_detection']
    transactions = db['transactions']

except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    raise HTTPException(status_code=500, detail="MongoDB connection failed")

# Root route for testing the MongoDB connection
@app.get("/")
def read_root():
    return {"message": "MongoDB Connection Test"}

# Route to simulate a cash transaction and detect fraud
@app.post("/simulate_transaction/")
async def simulate_transaction(cash_transaction: bool, invoice_issued: bool):
    try:
        timestamp = time.time()  # Record the time of the transaction
        fraud = False  # Initially, fraud is set to False

        # Check for fraud: if no invoice is issued and 15 seconds have passed
        if not invoice_issued and time.time() - timestamp > 15:
            fraud = True

        # Create a record for the transaction
        transaction_record = {
            "timestamp": timestamp,
            "cash_transaction": cash_transaction,
            "invoice_issued": invoice_issued,
            "fraud": fraud
        }

        # Insert the transaction record into MongoDB
        transactions.insert_one(transaction_record)
        
        # Fetch the last transaction to verify the insertion
        last_transaction = transactions.find_one(sort=[('_id', -1)])
        
        # Return the status, fraud status, and the last transaction
        return {"status": "Transaction Recorded", "fraud": fraud, "last_transaction": last_transaction}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")