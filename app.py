from flask import Flask, render_template, request, session, redirect, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# -------------------------------
# APP SETUP
# -------------------------------
app = Flask(__name__)
app.secret_key = "mysecret123"

# -------------------------------
# DATABASE SETUP
# -------------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# -------------------------------
# MODELS
# -------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), default="New Conversation")
    created_at = db.Column(db.DateTime, default=db.func.now())
    chats = db.relationship(
        'Chat',
        backref='conversation',
        lazy=True,
        cascade="all, delete-orphan"
    )


class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    query = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.now())

# -------------------------------
# LOAD ENV
# -------------------------------
load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

# -------------------------------
# RAG SETUP
# -------------------------------
embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="medicalbot",
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    max_output_tokens=500
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# -------------------------------
# HELPERS
# -------------------------------
def get_or_create_active_conversation(user_id):
    conv_id = session.get('active_conv_id')

    if conv_id:
        conv = Conversation.query.filter_by(id=conv_id, user_id=user_id).first()
        if conv:
            return conv

    conv = Conversation(user_id=user_id)
    db.session.add(conv)
    db.session.commit()

    session['active_conv_id'] = conv.id
    return conv


def conversations_for_user(user_id):
    return (
        Conversation.query
        .filter_by(user_id=user_id)
        .order_by(Conversation.created_at.desc())
        .all()
    )


def admin_required():
    return session.get('is_admin') is True

# -------------------------------
# AUTH ROUTES
# -------------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not username:
            return render_template('register.html', error="Username is required")

        if not email:
            return render_template('register.html', error="Email is required")

        if not password:
            return render_template('register.html', error="Password is required")

        if User.query.filter_by(email=email).first():
            return render_template('register.html', error="Email already exists")

        user = User(
            username=username,
            email=email,
            password=generate_password_hash(password)
        )

        db.session.add(user)
        db.session.commit()

        return redirect('/login?success=1')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    success = request.args.get('success')

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        # Admin login through same login page
        if email == "admin@gmail.com" and password == "admin":
            session.clear()
            session['is_admin'] = True
            session['admin_email'] = email
            return redirect('/admin/dashboard')

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session.clear()
            session['user_id'] = user.id
            session['username'] = user.username
            session['email'] = user.email
            session['is_admin'] = False
            return redirect('/chat')

        return render_template(
            'login.html',
            error="Invalid email or password",
            success=success
        )

    return render_template('login.html', success=success)


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# -------------------------------
# MAIN ROUTES
# -------------------------------
@app.route("/")
def home():
    if session.get('is_admin'):
        return redirect('/admin/dashboard')

    if 'user_id' in session:
        return redirect('/chat')

    return redirect('/login')


@app.route("/chat")
def chat_page():
    if 'user_id' not in session:
        return redirect('/login')

    user_id = session['user_id']
    conv = get_or_create_active_conversation(user_id)
    convs = conversations_for_user(user_id)

    return render_template(
        'chat.html',
        session_username=session.get('username', 'User'),
        session_email=session.get('email', ''),
        conversations=convs,
        active_conv_id=conv.id
    )

# -------------------------------
# CHAT API
# -------------------------------
@app.route("/get", methods=["POST"])
def chat():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    msg = request.form.get("msg", "").strip()
    conv_id = request.form.get("conv_id", type=int)
    user_id = session['user_id']

    if not msg:
        return jsonify({"error": "Empty message"}), 400

    conv = None

    if conv_id:
        conv = Conversation.query.filter_by(id=conv_id, user_id=user_id).first()

    if not conv:
        conv = get_or_create_active_conversation(user_id)

    try:
        response = rag_chain.invoke({"input": msg})
        print("RAG RESPONSE:", response)

        answer = response.get("answer")

        if not answer:
            answer = "Sorry, I could not generate a response."

        if conv.title == "New Conversation":
            conv.title = msg[:60] + ("..." if len(msg) > 60 else "")

        chat_entry = Chat(
            user_id=user_id,
            conversation_id=conv.id,
            query=msg,
            response=answer
        )

        db.session.add(chat_entry)
        db.session.commit()

        return jsonify({
            "answer": answer,
            "conv_id": conv.id
        })

    except Exception as e:
        db.session.rollback()
        print("RAG ERROR:", str(e))

        return jsonify({
            "answer": "Backend error: " + str(e)
        }), 500

# -------------------------------
# CONVERSATION API
# -------------------------------
@app.route("/new_conversation", methods=["POST"])
def new_conversation():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conv = Conversation(user_id=session['user_id'])
    db.session.add(conv)
    db.session.commit()

    session['active_conv_id'] = conv.id

    return jsonify({"conv_id": conv.id})


@app.route("/conversations", methods=["GET"])
def get_conversations():
    if 'user_id' not in session:
        return jsonify([]), 401

    convs = conversations_for_user(session['user_id'])

    return jsonify([{
        "id": c.id,
        "title": c.title,
        "date": c.created_at.strftime("%b %d")
    } for c in convs])


@app.route("/load_conversation/<int:conv_id>", methods=["GET"])
def load_conversation(conv_id):
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conv = Conversation.query.filter_by(
        id=conv_id,
        user_id=session['user_id']
    ).first()

    if not conv:
        return jsonify({"error": "Not found"}), 404

    session['active_conv_id'] = conv_id

    messages = [{
        "query": c.query,
        "response": c.response,
        "time": c.created_at.strftime("%H:%M")
    } for c in conv.chats]

    return jsonify({"messages": messages})


@app.route("/delete_conversation/<int:conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conv = Conversation.query.filter_by(
        id=conv_id,
        user_id=session['user_id']
    ).first()

    if not conv:
        return jsonify({"error": "Not found"}), 404

    db.session.delete(conv)
    db.session.commit()

    if session.get('active_conv_id') == conv_id:
        session.pop('active_conv_id', None)

    return jsonify({"success": True})

# -------------------------------
# ADMIN ROUTES
# -------------------------------
@app.route('/admin/dashboard')
def admin_dashboard():
    if not admin_required():
        return redirect('/login')

    total_users = db.session.query(User).count()
    total_conversations = db.session.query(Conversation).count()
    total_chats = db.session.query(Chat).count()
    users = db.session.query(User).order_by(User.id.desc()).all()

    return render_template(
        'admin_dashboard.html',
        total_users=total_users,
        total_conversations=total_conversations,
        total_chats=total_chats,
        users=users
    )


@app.route('/admin/user/<int:user_id>')
def admin_user_detail(user_id):
    if not admin_required():
        return redirect('/login')

    user = db.session.get(User, user_id)
    if not user:
        return "User not found", 404

    conversations = (
        Conversation.query
        .filter_by(user_id=user.id)
        .order_by(Conversation.created_at.desc())
        .all()
    )

    return render_template(
        'admin_user_detail.html',
        user=user,
        conversations=conversations
    )


@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    if not admin_required():
        return redirect('/login')

    user = db.session.get(User, user_id)
    if not user:
        return "User not found", 404

    db.session.query(Chat).filter_by(user_id=user.id).delete()
    db.session.query(Conversation).filter_by(user_id=user.id).delete()

    db.session.delete(user)
    db.session.commit()

    return redirect('/admin/dashboard')


@app.route('/admin/delete_conversation/<int:conv_id>', methods=['POST'])
def admin_delete_conversation(conv_id):
    if not admin_required():
        return redirect('/login')

    conv = db.session.get(Conversation, conv_id)
    if not conv:
        return "Conversation not found", 404
    user_id = conv.user_id

    db.session.delete(conv)
    db.session.commit()

    return redirect(f'/admin/user/{user_id}')


@app.route('/admin/logout')
def admin_logout():
    session.clear()
    return redirect('/login')

# -------------------------------
# CREATE DB
# -------------------------------
with app.app_context():
    db.create_all()

# -------------------------------
# RUN
# -------------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)