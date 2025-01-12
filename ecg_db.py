import sqlite3
import streamlit as st

def init_db():
    conn = sqlite3.connect("ecg_app.db")
    cursor = conn.cursor()

    # Create Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)

    # Create Results Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        result_name TEXT NOT NULL,
        result_data TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()

def verify_tables():
    conn = sqlite3.connect("ecg_app.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    print("Tables in Database:", tables)
    return tables

verify_tables()

# Ensure these functions are defined in ecg_db.py
def register_user(username, password):
    conn = sqlite3.connect("ecg_app.db")
    cursor = conn.cursor()

    # Check if the username already exists
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    existing_user = cursor.fetchone()

    if existing_user:
        conn.close()
        return False  # Username already exists

    # If the username doesn't exist, proceed with registration
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()
    return True  # User successfully registered
    


def authenticate_user(username, password):
    conn = sqlite3.connect("ecg_app.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user

def save_result(user_id, result_name, result_data):
    try:
        conn = sqlite3.connect("ecg_app.db")
        cursor = conn.cursor()
        
        # Insert data into the database
        cursor.execute(
            "INSERT INTO results (user_id, result_name, result_data) VALUES (?, ?, ?)",
            (user_id, result_name, result_data),
        )
        conn.commit()  # Commit the changes

        # Fetch all rows from the table for debugging purposes
        cursor.execute("SELECT * FROM results")
        rows = cursor.fetchall()

        # Print or display the current data in the table
        
    except Exception as e:
        st.error(f"Error saving result: {e}")
        print(f"Error saving result: {e}")
    finally:
        # Ensure the connection is closed
        conn.close()


def get_results(user_id):
    conn = sqlite3.connect("ecg_app.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results WHERE user_id = ?", (user_id,))
    results = cursor.fetchall()
    conn.close()
    return results

def update_result(result_id, new_name, new_data):
    conn = sqlite3.connect("ecg_app.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE results SET result_name = ?, result_data = ? WHERE id = ?", 
                   (new_name, new_data, result_id))
    conn.commit()
    conn.close()

def delete_result(result_id):
    conn = sqlite3.connect("ecg_app.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM results WHERE id = ?", (result_id,))
    conn.commit()
    conn.close()

