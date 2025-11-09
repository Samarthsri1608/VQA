import os
import sqlite3
import hashlib
import secrets
from functools import wraps
from flask import Blueprint, request, jsonify, session, redirect, send_file
from pathlib import Path

auth_bp = Blueprint('auth', __name__)

# Initialize SQLite database
DB_PATH = Path('users.db')
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

def hash_password(password: str) -> str:
    """Hash a password for storing."""
    salt = secrets.token_hex(8)
    pwdhash = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('ascii'), 
        100000
    )
    return f"{salt}${pwdhash.hex()}"

def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verify a stored password against one provided by user"""
    salt, stored_pwdhash = stored_password.split('$')
    pwdhash = hashlib.pbkdf2_hmac(
        'sha256',
        provided_password.encode('utf-8'),
        salt.encode('ascii'),
        100000
    )
    return pwdhash.hex() == stored_pwdhash

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.headers.get('Content-Type') == 'application/json':
                return jsonify({"error": "Authentication required"}), 401
            return redirect('/auth/login')
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/login', methods=['GET'])
def login_page():
    if 'user_id' in session:
        return redirect('/')
    return send_file('template/login.html')

@auth_bp.route('/register', methods=['GET'])
def register_page():
    if 'user_id' in session:
        return redirect('/')
    return send_file('template/register.html')

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute('SELECT id, password_hash FROM users WHERE email = ?', (email,))
            result = cur.fetchone()
            
            if not result or not verify_password(result[1], password):
                return jsonify({"error": "Invalid email or password"}), 401
            
            session['user_id'] = result[0]
            return jsonify({"message": "Login successful"})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    if not all([name, email, password]):
        return jsonify({"error": "All fields are required"}), 400
    
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            # Check if email already exists
            cur.execute('SELECT id FROM users WHERE email = ?', (email,))
            if cur.fetchone():
                return jsonify({"error": "Email already registered"}), 409
                
            # Create new user
            password_hash = hash_password(password)
            cur.execute(
                'INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)',
                (name, email, password_hash)
            )
            return jsonify({"message": "Registration successful"})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect('/auth/login')