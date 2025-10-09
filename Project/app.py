# app.py (FULL updated file with Step 2 & 3: admin user management + aspect CRUD)
from flask import Flask, render_template, redirect, url_for, request, flash, abort, Response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import pandas as pd
import io
import json
import re
from collections import Counter

# IMPORT from your sentiment module (must be present)
# analyze_sentiment(text) -> (sentiment_string, preprocessed_text, aspect)
# preprocess(text) -> str
# POS_WORDS / NEG_WORDS sets present in sentiment.py help prioritize top words
from sentiment import analyze_sentiment, preprocess, POS_WORDS, NEG_WORDS, predict_proba

app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace_this_with_a_random_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv', 'txt'}

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# ---------- Models ----------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='user')
    datasets = db.relationship('Dataset', backref='owner', lazy=True)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    active = db.Column(db.Boolean, default=True)


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_name = db.Column(db.String(200))
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    reviews = db.relationship('Review', backref='dataset', lazy=True, cascade="all, delete-orphan")


class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    raw_text = db.Column(db.Text, nullable=False)
    preprocessed_text = db.Column(db.Text)
    sentiment = db.Column(db.String(20))
    aspect = db.Column(db.String(120))
    confidence = db.Column(db.Float)   # 🆕 Added this line for confidence %
    date = db.Column(db.DateTime)


class Aspect(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    industry = db.Column(db.String(120))


class Log(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    action_type = db.Column(db.String(200))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------- Utility helpers ----------
STOPWORDS = set([
    "the", "and", "for", "with", "that", "this", "has", "have", "are", "was",
    "were", "but", "not", "you", "your", "from", "they", "their", "will", "can",
    "its", "it's", "all", "one", "very", "also", "we", "our", "they", "them",
    "been", "which", "when", "what", "how", "out", "new", "use", "review", "product"
])


def create_log(action, user_id=None):
    """Helper to create recent activity log (commits immediately)."""
    try:
        log = Log(action_type=action, user_id=user_id)
        db.session.add(log)
        db.session.commit()
    except Exception:
        db.session.rollback()


def top_words_by_sentiment(reviews, sentiment_label, top_n=5):
    """
    Count words that appear in reviews with sentiment `sentiment_label`.
    Preference is given to words from POS_WORDS/NEG_WORDS; otherwise returns
    most common words (excluding STOPWORDS).
    Returns list of tuples: [(word, count), ...]
    """
    cnt = Counter()
    for r in reviews:
        if (r.get("sentiment") or "").strip() != sentiment_label:
            continue
        pre = r.get("preprocessed_text") or preprocess(r.get("raw_text") or "")
        words = re.findall(r'\b[a-zA-Z]{3,}\b', pre)
        for w in words:
            w = w.lower()
            if w in STOPWORDS:
                continue
            cnt[w] += 1

    # Prioritize known sentiment lexicon membership
    prioritized = []
    if sentiment_label == "Positive":
        prioritized = [(w, c) for w, c in cnt.items() if w in POS_WORDS]
    elif sentiment_label == "Negative":
        prioritized = [(w, c) for w, c in cnt.items() if w in NEG_WORDS]

    if prioritized:
        prioritized = sorted(prioritized, key=lambda x: x[1], reverse=True)[:top_n]
        # If less than top_n prioritized words, fill from general counts
        if len(prioritized) < top_n:
            needed = top_n - len(prioritized)
            remaining = [(w, c) for w, c in cnt.most_common(top_n + needed) if (w, c) not in prioritized]
            prioritized.extend(remaining[:needed])
        return prioritized[:top_n]

    # fallback: most common words in those sentiment reviews
    return cnt.most_common(top_n)


# ---------- Routes ----------
@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    # if already logged in, redirect to user's dashboard
    if current_user.is_authenticated:
        return redirect(url_for('dashboard' if current_user.role == 'user' else 'admin_dashboard'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Simple server-side validation (keeps it relaxed per your request)
        if not name or not email or not password or not confirm_password:
            flash('Please fill all fields.', 'danger')
            return render_template('user/register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('user/register.html')

        # Basic email sanity check
        if '@' not in email or email.startswith('@') or email.endswith('@'):
            flash('Enter a valid email address.', 'danger')
            return render_template('user/register.html')

        # prevent duplicate registrations
        if User.query.filter_by(email=email).first():
            flash('Email already registered. Try logging in.', 'warning')
            return render_template('user/register.html')

        # All good -> create user (hash password)
        pw_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(name=name, email=email, password=pw_hash, role='user')
        db.session.add(user)
        db.session.commit()

        # optional: write a log entry if you have create_log helper
        try:
            create_log(f"New user registered: {name}", user.id)
        except Exception:
            pass

        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))

    # GET -> show form
    return render_template('user/register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard' if current_user.role == 'user' else 'admin_dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()

        # Fix: remove user.active check
        if user and bcrypt.check_password_hash(user.password, password):
            # Update last_active
            user.last_active = datetime.utcnow()
            db.session.commit()

            login_user(user)
            create_log(f"User {user.name} logged in", user.id)
            flash('Logged in successfully', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard' if user.role == 'user' else 'admin_dashboard'))

        flash('Login failed — check email/password', 'danger')

    return render_template('user/login.html')

# ---------------- Admin: User Management ---------------- #
@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    users = User.query.all()
    return render_template('admin/users.html', users=users)


@app.route('/admin/users/<int:user_id>/view')
@login_required
def admin_view_user(user_id):
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    
    user = User.query.get_or_404(user_id)
    datasets = Dataset.query.filter_by(user_id=user.id).all()
    
    # if you have a Report model
    try:
        reports = Report.query.filter_by(user_id=user.id).all()
    except Exception:
        reports = []  # fallback if no Report table exists
    
    return render_template(
        'admin/user_profile.html',
        user=user,
        datasets=datasets,
        reports=reports
    )


@app.route('/admin/users/<int:user_id>/deactivate')
@login_required
def admin_deactivate_user(user_id):
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    
    user = User.query.get_or_404(user_id)
    user.active = False
    db.session.commit()
    create_log(f"Admin deactivated user {user.name}", current_user.id)
    flash(f"User {user.name} deactivated.", "warning")
    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/delete')
@login_required
def admin_delete_user(user_id):
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    create_log(f"Admin deleted user {user.name}", current_user.id)
    flash(f"User {user.name} deleted.", "danger")
    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/reset_password')
@login_required
def admin_reset_password(user_id):
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    
    user = User.query.get_or_404(user_id)
    # Reset password to a default (you can change logic)
    default_pw = "User@123"
    user.password = bcrypt.generate_password_hash(default_pw).decode('utf-8')
    db.session.commit()
    create_log(f"Admin reset password for user {user.name}", current_user.id)
    flash(f"Password for {user.name} reset to {default_pw}", "info")
    return redirect(url_for('admin_users'))

@app.route('/logout')
@login_required
def logout():
    user_role = current_user.role
    try:
        create_log(f"{current_user.name} logged out", current_user.id)
    except Exception:
        pass

    logout_user()
    # If admin → go to landing page, else → login page
    if user_role == 'admin':
        return redirect(url_for('index'))
    return redirect(url_for('login'))

# ---------- Dashboard (file upload + manual raw review) ----------
# ---------- Dashboard (file upload + manual raw review) ----------
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if current_user.role != 'user':
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        # 1) File upload handling
        uploaded_file = request.files.get('file')
        if uploaded_file and uploaded_file.filename:
            filename = secure_filename(uploaded_file.filename)
            if not allowed_file(filename):
                flash('Only CSV or TXT files are allowed.', 'danger')
                return redirect(url_for('dashboard'))
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(path)

            ds = Dataset(user_id=current_user.id, file_name=filename)
            db.session.add(ds)
            db.session.commit()

            inserted = 0
            try:
                if filename.lower().endswith('.csv'):
                    df = None
                    try:
                        df = pd.read_csv(path)
                    except Exception:
                        try:
                            df = pd.read_csv(path, engine='python', on_bad_lines='skip')
                        except Exception:
                            df = None

                    if df is None:
                        with open(path, encoding='utf-8', errors='replace') as f:
                            lines = [line.strip() for line in f if line.strip()]
                        if lines and any(h in lines[0].lower() for h in ['review', 'text', 'content', 'comment']):
                            lines = lines[1:]
                        for line in lines:
                            if line and line.lower() not in ['nan', 'none', 'null']:
                                sentiment, preprocessed, aspect = analyze_sentiment(line)
                                proba = predict_proba(line)
                                confidence = round(proba.get(sentiment, 0.0) * 100, 2)
                                r = Review(
                                    dataset_id=ds.id,
                                    raw_text=line,
                                    preprocessed_text=preprocessed,
                                    sentiment=sentiment,
                                    aspect=aspect,
                                    confidence=confidence,
                                    date=datetime.utcnow()
                                )
                                db.session.add(r)
                                inserted += 1
                    else:
                        df.columns = [str(c).strip().lower() for c in df.columns]
                        series = None
                        for candidate in ['review', 'text', 'content', 'comment']:
                            if candidate in df.columns:
                                series = df[candidate].astype(str)
                                break
                        if series is None:
                            if df.shape[1] == 1:
                                series = df.iloc[:, 0].astype(str)
                            else:
                                series = df.astype(str).apply(
                                    lambda row: ', '.join([str(x) for x in row if str(x).strip().lower() not in ['nan', 'none', 'null']]),
                                    axis=1)
                        for raw in series:
                            raw = str(raw).strip()
                            if raw and raw.lower() not in ['nan', 'none', 'null']:
                                sentiment, preprocessed, aspect = analyze_sentiment(raw)
                                proba = predict_proba(raw)
                                confidence = round(proba.get(sentiment, 0.0) * 100, 2)
                                r = Review(
                                    dataset_id=ds.id,
                                    raw_text=raw,
                                    preprocessed_text=preprocessed,
                                    sentiment=sentiment,
                                    aspect=aspect,
                                    confidence=confidence,
                                    date=datetime.utcnow()
                                )
                                db.session.add(r)
                                inserted += 1
                else:
                    with open(path, encoding='utf-8', errors='replace') as f:
                        for line in f:
                            l = line.strip()
                            if l and l.lower() not in ['nan', 'none', 'null']:
                                sentiment, preprocessed, aspect = analyze_sentiment(l)
                                proba = predict_proba(l)
                                confidence = round(proba.get(sentiment, 0.0) * 100, 2)
                                r = Review(
                                    dataset_id=ds.id,
                                    raw_text=l,
                                    preprocessed_text=preprocessed,
                                    sentiment=sentiment,
                                    aspect=aspect,
                                    confidence=confidence,
                                    date=datetime.utcnow()
                                )
                                db.session.add(r)
                                inserted += 1

                db.session.commit()
                create_log(f"User {current_user.name} uploaded dataset '{filename}' — {inserted} reviews analyzed & saved", current_user.id)
                flash(f'Upload saved: {inserted} reviews analyzed and added to dataset "{filename}".', 'success')
            except Exception as e:
                app.logger.exception("Error parsing uploaded file: %s", str(e))
                flash('Upload succeeded but parsing failed. Please check file format.', 'warning')

            return redirect(url_for('dashboard'))

        # 2) Manual raw review handling
        raw_review = request.form.get('raw_review', '').strip()
        if raw_review:
            ds = Dataset(user_id=current_user.id, file_name="manual_entry")
            db.session.add(ds)
            db.session.commit()

            sentiment, preprocessed, aspect = analyze_sentiment(raw_review)
            proba = predict_proba(raw_review)
            confidence = round(proba.get(sentiment, 0.0) * 100, 2)

            r = Review(
                dataset_id=ds.id,
                raw_text=raw_review,
                preprocessed_text=preprocessed,
                sentiment=sentiment,
                aspect=aspect,
                confidence=confidence,
                date=datetime.utcnow()
            )
            db.session.add(r)
            db.session.commit()
            create_log(f"User {current_user.name} added manual review — sentiment: {sentiment}", current_user.id)
            flash('Manual review added and analyzed automatically.', 'success')
            return redirect(url_for('dashboard'))

        flash('Please upload a file or type a review before submitting.', 'warning')
        return redirect(url_for('dashboard'))

    # ---------- GET: show dashboard ----------
    from sqlalchemy import func

    user_id = current_user.id
    datasets_count = Dataset.query.filter_by(user_id=user_id).count()
    reviews_count = Review.query.join(Dataset).filter(Dataset.user_id == user_id).count()
    uploads = Dataset.query.filter_by(user_id=user_id).order_by(Dataset.upload_date.desc()).limit(5).all()

    # 1) Backfill confidence for this user's reviews that still have NULL
    missing = (Review.query
               .join(Dataset)
               .filter(Dataset.user_id == user_id, Review.confidence.is_(None))
               .all())
    if missing:
        for r in missing:
            try:
                s, pre, asp = analyze_sentiment(r.raw_text or "")
                probs = predict_proba(r.raw_text or "")
                label = s or max(probs, key=probs.get)
                r.preprocessed_text = pre
                r.sentiment = s
                r.aspect = asp
                r.confidence = round(float(probs.get(label, 0.0)) * 100.0, 2)
                if not r.date:
                    r.date = datetime.utcnow()
            except Exception:
                # If anything goes wrong, at least make it 0 so it won't block averaging
                r.confidence = 0.0
        db.session.commit()

    # 2) Compute average confidence (values already stored as 0–100)
    avg_conf = (db.session.query(func.avg(Review.confidence))
                .join(Dataset)
                .filter(Dataset.user_id == user_id, Review.confidence.isnot(None))
                .scalar()) or 0.0
    avg_confidence = round(avg_conf, 2)

    return render_template(
        'user/dashboard.html',
        datasets_count=datasets_count,
        reviews_count=reviews_count,
        uploads=uploads,
        avg_confidence=avg_confidence
    )

@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        abort(403)
    users_count = User.query.count()
    datasets_count = Dataset.query.count()
    reviews_count = Review.query.count()
    recent_logs = Log.query.order_by(Log.timestamp.desc()).limit(10).all()
    # you can compute active_users_today here (placeholder)
    active_users_today = User.query.filter(User.last_active >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)).count()
    return render_template('admin/dashboard.html', users_count=users_count, datasets_count=datasets_count, reviews_count=reviews_count, recent_logs=recent_logs, active_users_today=active_users_today)
# ---------------- Admin: Aspect management ---------------- #
from collections import Counter

@app.route('/admin/aspects')
@login_required
def admin_aspects():
    if current_user.role != 'admin':
        abort(403)

    # Load all reviews (admin view)
    reviews_q = Review.query.all()

    # Aggregate by aspect name
    aspect_map = {}
    for r in reviews_q:
        a = (r.aspect or "General").strip()
        if not a:
            a = "General"
        # keep aspect display case as stored
        if a not in aspect_map:
            aspect_map[a] = {"name": a, "positive": 0, "negative": 0, "neutral": 0, "texts": []}

        s = (r.sentiment or "").strip()
        if s == "Positive":
            aspect_map[a]["positive"] += 1
        elif s == "Negative":
            aspect_map[a]["negative"] += 1
        else:
            aspect_map[a]["neutral"] += 1

        # prefer preprocessed text for top-words; fallback to raw_text
        text_for_words = r.preprocessed_text or r.raw_text or ""
        aspect_map[a]["texts"].append(text_for_words)

    # helper: compute top words for a list of texts
    def top_words_from_texts(texts, top_n=4):
        cnt = Counter()
        for t in texts:
            try:
                # use your preprocess if available to normalize words
                norm = preprocess(t) if callable(preprocess) else t
            except Exception:
                norm = t or ""
            words = re.findall(r'\b[a-zA-Z]{3,}\b', (norm or "").lower())
            for w in words:
                if w in STOPWORDS:
                    continue
                cnt[w] += 1
        return [w for w, _ in cnt.most_common(top_n)]

    # Build three lists for template
    positive_aspects = []
    negative_aspects = []
    neutral_aspects = []

    for name, info in aspect_map.items():
        info["count_total"] = info["positive"] + info["negative"] + info["neutral"]
        # top words
        info["top_words"] = top_words_from_texts(info["texts"], top_n=4)
        # Append into appropriate buckets if they have >0 count
        if info["positive"] > 0:
            positive_aspects.append(info)
        if info["negative"] > 0:
            negative_aspects.append(info)
        if info["neutral"] > 0:
            neutral_aspects.append(info)

    # sort each bucket (by the sentiment count so most relevant show first)
    positive_aspects.sort(key=lambda x: x["positive"], reverse=True)
    negative_aspects.sort(key=lambda x: x["negative"], reverse=True)
    neutral_aspects.sort(key=lambda x: x["neutral"], reverse=True)

    return render_template(
        'admin/aspects.html',
        positive_aspects=positive_aspects,
        negative_aspects=negative_aspects,
        neutral_aspects=neutral_aspects
    )

@app.route('/admin/aspects/add', methods=['POST'])
@login_required
def admin_aspect_add():
    if current_user.role != 'admin':
        abort(403)

    name = request.form.get('name', '').strip()
    if not name:
        flash('Aspect name required', 'danger')
        return redirect(url_for('admin_aspects'))

    a = Aspect(name=name)
    db.session.add(a)
    db.session.commit()
    create_log(f"Admin added aspect '{name}'", current_user.id)
    flash('Aspect added.', 'success')
    return redirect(url_for('admin_aspects'))


@app.route('/admin/aspects/edit/<int:aspect_id>', methods=['POST'])
@login_required
def admin_aspect_edit(aspect_id):
    if current_user.role != 'admin':
        abort(403)

    aspect = Aspect.query.get_or_404(aspect_id)
    name = request.form.get('name', '').strip()
    if not name:
        flash('Aspect name required', 'danger')
        return redirect(url_for('admin_aspects'))

    aspect.name = name
    db.session.commit()
    create_log(f"Admin edited aspect '{name}'", current_user.id)
    flash('Aspect updated.', 'success')
    return redirect(url_for('admin_aspects'))


@app.route('/admin/aspects/delete/<int:aspect_id>', methods=['POST'])
@login_required
def admin_aspect_delete(aspect_id):
    if current_user.role != 'admin':
        abort(403)

    aspect = Aspect.query.get_or_404(aspect_id)
    try:
        db.session.delete(aspect)
        db.session.commit()
        create_log(f"Admin deleted aspect '{aspect.name}'", current_user.id)
        flash('Aspect deleted.', 'success')
    except Exception as e:
        db.session.rollback()
        app.logger.exception("Error deleting aspect: %s", e)
        flash('Error deleting aspect.', 'danger')

    return redirect(url_for('admin_aspects'))
# ---------------- Admin: System Monitoring ---------------- #
@app.route('/admin/monitor')
@login_required
def admin_monitor():
    if current_user.role != 'admin':
        abort(403)

    # 1. Recent activity logs
    recent_logs = Log.query.order_by(Log.timestamp.desc()).limit(20).all()

    # 2. Performance logs → fetch datasets processed recently
    performance_logs = []
    datasets = Dataset.query.order_by(Dataset.upload_date.desc()).limit(5).all()
    for ds in datasets:
        reviews_count = Review.query.filter_by(dataset_id=ds.id).count()
        # For demo: pretend processing time = reviews_count / 50 (approx seconds)
        perf_time = round(reviews_count / 50.0, 2) if reviews_count else 0
        performance_logs.append({
            "file_name": ds.file_name,
            "reviews": reviews_count,
            "time": perf_time,
            "date": ds.upload_date.strftime("%Y-%m-%d %H:%M")
        })

    # 3. Model accuracy check → sample 5 reviews
    sample_reviews = Review.query.order_by(Review.date.desc()).limit(5).all()

    # 4. System stats (datasets, reviews, users)
    users_count = User.query.count()
    datasets_count = Dataset.query.count()
    reviews_count = Review.query.count()

    return render_template(
        'admin/monitor.html',
        recent_logs=recent_logs,
        performance_logs=performance_logs,
        sample_reviews=sample_reviews,
        users_count=users_count,
        datasets_count=datasets_count,
        reviews_count=reviews_count
    )

@app.route('/admin/reports')
@login_required
def admin_reports():
    if current_user.role != 'admin':
        abort(403)

    # Stats
    users_count = User.query.count()
    datasets_count = Dataset.query.count()
    reviews_count = Review.query.count()

    # Daily reviews
    today = datetime.utcnow().date()
    daily_reviews = Review.query.filter(
        Review.date >= datetime(today.year, today.month, today.day)
    ).count()

    # Most active users (by datasets & reviews)
    active_users = []
    users = User.query.all()
    for u in users:
        datasets = Dataset.query.filter_by(user_id=u.id).count()
        reviews = Review.query.join(Dataset).filter(Dataset.user_id == u.id).count()
        if datasets or reviews:
            active_users.append({
                "name": u.name,
                "email": u.email,
                "datasets": datasets,
                "reviews": reviews
            })
    active_users = sorted(active_users, key=lambda x: x["reviews"], reverse=True)[:5]

    # Common aspects
    aspect_map = {}
    for r in Review.query.all():
        asp = r.aspect or "General"
        if asp not in aspect_map:
            aspect_map[asp] = {"name": asp, "positive": 0, "negative": 0, "neutral": 0}
        if r.sentiment == "Positive":
            aspect_map[asp]["positive"] += 1
        elif r.sentiment == "Negative":
            aspect_map[asp]["negative"] += 1
        else:
            aspect_map[asp]["neutral"] += 1
    common_aspects = sorted(aspect_map.values(), key=lambda x: (x["positive"]+x["negative"]+x["neutral"]), reverse=True)[:5]

    return render_template(
        'admin/reports.html',
        users_count=users_count,
        datasets_count=datasets_count,
        reviews_count=reviews_count,
        daily_reviews=daily_reviews,
        active_users=active_users,
        common_aspects=common_aspects
    )

# ---------------- Admin: Insights & Visuals (ALL users) ---------------- #
@app.route('/admin/insights')
@login_required
def admin_insights():
    if current_user.role != 'admin':
        abort(403)

    # Pull ALL reviews across ALL users
    reviews_q = Review.query.order_by(Review.id.asc()).all()

    # Serialize (optional, only if you want to inspect in template/JS)
    reviews = []
    for r in reviews_q:
        reviews.append({
            "id": r.id,
            "raw_text": (r.raw_text or ""),
            "preprocessed_text": r.preprocessed_text or "",
            "sentiment": r.sentiment or "",
            "aspect": (r.aspect or "General"),
            "date": r.date.isoformat() if isinstance(r.date, datetime) else (str(r.date) if r.date else "")
        })

    # Sentiment counts
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for r in reviews:
        s = (r.get("sentiment") or "").strip()
        if s in sentiment_counts:
            sentiment_counts[s] += 1

    # Aspect stats (count pos/neg/neu per aspect)
    aspect_map = {}
    for r in reviews:
        asp = r.get("aspect") or "General"
        if asp not in aspect_map:
            aspect_map[asp] = {"aspect": asp, "positive": 0, "negative": 0, "neutral": 0, "total": 0}
        s = r.get("sentiment")
        if s == "Positive":
            aspect_map[asp]["positive"] += 1
        elif s == "Negative":
            aspect_map[asp]["negative"] += 1
        else:
            aspect_map[asp]["neutral"] += 1
        aspect_map[asp]["total"] += 1

    aspect_stats_all = list(aspect_map.values())
    # Keep top 10 aspects by total count so the bar chart is readable
    aspect_stats = sorted(aspect_stats_all, key=lambda x: x["total"], reverse=True)[:10]

    # Trend over time (YYYY-MM-DD)
    grouped = {}
    for r in reviews_q:
        if r.date:
            key = r.date.strftime("%Y-%m-%d")
        else:
            continue
        if key not in grouped:
            grouped[key] = {"Positive": 0, "Negative": 0, "Neutral": 0}
        if r.sentiment == "Positive":
            grouped[key]["Positive"] += 1
        elif r.sentiment == "Negative":
            grouped[key]["Negative"] += 1
        else:
            grouped[key]["Neutral"] += 1

    try:
        date_keys = sorted(grouped.keys(), key=lambda x: datetime.strptime(x, "%Y-%m-%d")) if grouped else []
    except Exception:
        date_keys = sorted(grouped.keys()) if grouped else []

    trend_labels = date_keys
    trend_pos_values = [grouped[k]["Positive"] for k in date_keys]
    trend_neg_values = [grouped[k]["Negative"] for k in date_keys]
    trend_neu_values = [grouped[k]["Neutral"] for k in date_keys]

    # Small header stats
    users_count = User.query.count()
    datasets_count = Dataset.query.count()
    reviews_count = len(reviews_q)

    return render_template(
        'admin/insights.html',
        users_count=users_count,
        datasets_count=datasets_count,
        reviews_count=reviews_count,
        sentiment_counts=sentiment_counts,
        aspect_stats=aspect_stats,
        trend_labels=trend_labels,
        trend_pos_values=trend_pos_values,
        trend_neg_values=trend_neg_values,
        trend_neu_values=trend_neu_values
    )

# --------- view_reviews route (show rows for a dataset) ----------
@app.route('/reviews/<int:dataset_id>')
@login_required
def view_reviews(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != current_user.id and current_user.role != 'admin':
        abort(403)
    reviews = Review.query.filter_by(dataset_id=dataset_id).all()
    return render_template('user/reviews.html', dataset=dataset, reviews=reviews)


@app.route('/view_reviews/<int:dataset_id>')
@login_required
def view_reviews_alias(dataset_id):
    # keep compatibility
    return redirect(url_for('view_reviews', dataset_id=dataset_id))


# ---------- Helper shortcut ----------
@app.route('/analysis')
@login_required
def analysis_root():
    if current_user.role == 'admin':
        ds = Dataset.query.order_by(Dataset.upload_date.desc()).first()
    else:
        ds = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.upload_date.desc()).first()
    if not ds:
        flash('No datasets found — please upload first.', 'warning')
        return redirect(url_for('dashboard'))
    return redirect(url_for('analysis', dataset_id=ds.id))

# ---------- Analysis for a single dataset ----------
@app.route('/analysis/<int:dataset_id>')
@login_required
def dataset_analysis(dataset_id):   # use only ONE function
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != current_user.id and current_user.role != 'admin':
        abort(403)

    reviews_q = Review.query.filter_by(dataset_id=dataset.id).all()

    # serialize reviews
    reviews = []
    for r in reviews_q:
        reviews.append({
            "id": r.id,
            "raw_text": (r.raw_text or "") if not isinstance(r.raw_text, bytes) else r.raw_text.decode("utf-8", errors="ignore"),
            "preprocessed_text": r.preprocessed_text or "",
            "sentiment": r.sentiment or "",
            "aspect": r.aspect or "General",
            "date": r.date.isoformat() if isinstance(r.date, datetime) else (str(r.date) if r.date else "")
        })

    # 🔹 NEW BLOCK — Confidence % enrichment
    # -------------------------------------------------------------
    # This calculates confidence for each review using predict_proba()
    # from sentiment.py (Logistic Regression probability output)
    # -------------------------------------------------------------
    for r in reviews:
        text = r.get("preprocessed_text") or preprocess(r.get("raw_text") or "")
        try:
            probs = predict_proba(text)
            label = (r.get("sentiment") or "").strip() or max(probs, key=probs.get)
            r["confidence"] = round(float(probs.get(label, max(probs.values()))) * 100, 2)
        except Exception:
            r["confidence"] = None
    # -------------------------------------------------------------

    # sentiment counts
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for r in reviews:
        s = (r.get("sentiment") or "").strip()
        if s in sentiment_counts:
            sentiment_counts[s] += 1

    # aspect stats
    aspect_map = {}
    for r in reviews:
        asp = (r.get("aspect") or "General")
        if asp not in aspect_map:
            aspect_map[asp] = {"aspect": asp, "positive": 0, "negative": 0, "neutral": 0}
        s = r.get("sentiment")
        if s == "Positive":
            aspect_map[asp]["positive"] += 1
        elif s == "Negative":
            aspect_map[asp]["negative"] += 1
        else:
            aspect_map[asp]["neutral"] += 1
    aspect_stats = list(aspect_map.values())

    # top aspects
    top_positive_aspects = sorted(aspect_stats, key=lambda x: x["positive"], reverse=True)[:5]
    top_negative_aspects = sorted(aspect_stats, key=lambda x: x["negative"], reverse=True)[:5]

    # top words
    top_positive_words = top_words_by_sentiment(reviews, "Positive", top_n=5)
    top_negative_words = top_words_by_sentiment(reviews, "Negative", top_n=5)

    # trend data
    grouped = {}
    for r in reviews_q:
        if r.date:
            key = r.date.strftime("%Y-%m-%d")
        else:
            key = dataset.upload_date.strftime("%Y-%m-%d") if dataset.upload_date else None
        if not key:
            continue
        if key not in grouped:
            grouped[key] = {"Positive": 0, "Negative": 0, "Neutral": 0}
        if r.sentiment == "Positive":
            grouped[key]["Positive"] += 1
        elif r.sentiment == "Negative":
            grouped[key]["Negative"] += 1
        else:
            grouped[key]["Neutral"] += 1

    try:
        date_keys = sorted(grouped.keys(), key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
    except Exception:
        date_keys = sorted(grouped.keys())

    trend_labels = date_keys
    trend_pos_values = [grouped[k]["Positive"] for k in date_keys]
    trend_neg_values = [grouped[k]["Negative"] for k in date_keys]
    trend_neu_values = [grouped[k]["Neutral"] for k in date_keys]
    trend_values = [trend_pos_values[i] + trend_neg_values[i] + trend_neu_values[i] for i in range(len(date_keys))]

    # ✅ FIX: no run_analysis anymore
    analyze_url = url_for('run_analysis_all')  
    export_csv_url = url_for('export_csv', dataset_id=dataset.id)
    export_pdf_url = url_for('export_pdf', dataset_id=dataset.id)

    return render_template("analysis.html",
                           dataset=dataset,
                           reviews=reviews,
                           sentiment_counts=sentiment_counts,
                           aspect_stats=aspect_stats,
                           top_positive_aspects=[(a["aspect"], a["positive"]) for a in top_positive_aspects],
                           top_negative_aspects=[(a["aspect"], a["negative"]) for a in top_negative_aspects],
                           top_positive_words=top_positive_words,
                           top_negative_words=top_negative_words,
                           trend_labels=trend_labels,
                           trend_values=trend_values,
                           trend_pos_values=trend_pos_values,
                           trend_neg_values=trend_neg_values,
                           trend_neu_values=trend_neu_values,
                           analyze_url=analyze_url,
                           export_csv_url=export_csv_url,
                           export_pdf_url=export_pdf_url)

# ---------- Analysis across ALL datasets ----------
# ---------- Analysis across ALL datasets ----------
@app.route('/analysis_all')
@login_required
def analysis_all():
    if current_user.role == 'admin':
        reviews_q = Review.query.order_by(Review.id.asc()).all()
    else:
        reviews_q = (
            Review.query.join(Dataset)
            .filter(Dataset.user_id == current_user.id)
            .order_by(Review.id.asc())
            .all()
        )

    # Convert reviews to a list of dictionaries
    reviews = []
    for r in reviews_q:
        reviews.append({
            "id": r.id,
            "raw_text": r.raw_text or "",
            "preprocessed_text": r.preprocessed_text or "",
            "sentiment": r.sentiment or "",
            "aspect": r.aspect or "General",
            "date": r.date.isoformat() if isinstance(r.date, datetime) else (str(r.date) if r.date else "")
        })

    # 🔹 Add confidence %
    for r in reviews:
        text = r.get("preprocessed_text") or preprocess(r.get("raw_text") or "")
        try:
            probs = predict_proba(text)
            label = (r.get("sentiment") or "").strip() or max(probs, key=probs.get)
            r["confidence"] = round(float(probs.get(label, max(probs.values()))) * 100, 2)
        except Exception:
            r["confidence"] = None

    # 🔹 Sentiment counts
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for r in reviews:
        s = (r.get("sentiment") or "").strip()
        if s in sentiment_counts:
            sentiment_counts[s] += 1

    # 🔹 Aspect stats
    aspect_map = {}
    for r in reviews:
        asp = (r.get("aspect") or "General")
        if asp not in aspect_map:
            aspect_map[asp] = {"aspect": asp, "positive": 0, "negative": 0, "neutral": 0}
        s = r.get("sentiment")
        if s == "Positive":
            aspect_map[asp]["positive"] += 1
        elif s == "Negative":
            aspect_map[asp]["negative"] += 1
        else:
            aspect_map[asp]["neutral"] += 1
    aspect_stats = list(aspect_map.values())

    # 🔹 Top words
    top_positive_words = top_words_by_sentiment(reviews, "Positive", top_n=5)
    top_negative_words = top_words_by_sentiment(reviews, "Negative", top_n=5)

    # 🔹 Trend data
    grouped = {}
    for r in reviews_q:
        if r.date:
            key = r.date.strftime("%Y-%m-%d")
        else:
            continue
        if key not in grouped:
            grouped[key] = {"Positive": 0, "Negative": 0, "Neutral": 0}
        if r.sentiment == "Positive":
            grouped[key]["Positive"] += 1
        elif r.sentiment == "Negative":
            grouped[key]["Negative"] += 1
        else:
            grouped[key]["Neutral"] += 1

    if grouped:
        date_keys = sorted(grouped.keys(), key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
    else:
        date_keys = []

    trend_labels = date_keys
    trend_pos_values = [grouped[k]["Positive"] for k in date_keys]
    trend_neg_values = [grouped[k]["Negative"] for k in date_keys]
    trend_neu_values = [grouped[k]["Neutral"] for k in date_keys]
    trend_values = [
        trend_pos_values[i] + trend_neg_values[i] + trend_neu_values[i]
        for i in range(len(date_keys))
    ]

    # ✅ Now send ALL variables required for chart.js and analysis.html
    return render_template(
        "analysis.html",
        dataset=None,
        reviews=reviews,
        sentiment_counts=sentiment_counts,
        aspect_stats=aspect_stats,
        top_positive_words=top_positive_words,
        top_negative_words=top_negative_words,
        trend_labels=trend_labels,
        trend_values=trend_values,              # 🧠 ADDED
        trend_pos_values=trend_pos_values,
        trend_neg_values=trend_neg_values,
        trend_neu_values=trend_neu_values,
        analyze_url=url_for('run_analysis_all'),
        export_csv_url=url_for('export_csv_all'),
        export_pdf_url=url_for('export_pdf_all')
    )

# ---------- Run analysis for ALL reviews ----------
@app.route('/run_analysis_all')
@login_required
def run_analysis_all():
    if current_user.role == 'admin':
        reviews_q = Review.query.order_by(Review.id.asc()).all()
    else:
        reviews_q = (
            Review.query.join(Dataset)
            .filter(Dataset.user_id == current_user.id)
            .order_by(Review.id.asc())
            .all()
        )

    updated = False

    for r in reviews_q:
        try:
            # Run sentiment + confidence
            sentiment, preprocessed, aspect = analyze_sentiment(r.raw_text)
            proba = predict_proba(r.raw_text)

            # pick probability of that sentiment
            confidence_score = round(float(proba.get(sentiment, 0.0)) * 100, 2)

            # Update DB fields
            r.preprocessed_text = preprocessed
            r.sentiment = sentiment
            r.aspect = aspect
            r.confidence = confidence_score  # ✅ now used in dashboard
            if not r.date:
                r.date = datetime.utcnow()

            updated = True

        except Exception as e:
            app.logger.exception(
                "run_analysis_all error for review %s: %s", r.id, str(e)
            )

    if updated:
        db.session.commit()
        create_log("Analysis completed for ALL reviews", current_user.id)
        flash("✅ Analysis completed successfully for all reviews.", "success")
    else:
        flash("ℹ️ No reviews required re-analysis.", "info")

    return redirect(url_for('analysis_all'))

# ---------- Export single-dataset CSV/PDF ----------
@app.route('/export_csv/<int:dataset_id>')
@login_required
def export_csv(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != current_user.id and current_user.role != 'admin':
        abort(403)
    reviews = Review.query.filter_by(dataset_id=dataset.id).all()
    output = io.StringIO()
    output.write("raw_text,preprocessed_text,sentiment,aspect,date\n")
    for r in reviews:
        raw = (r.raw_text or "").replace('"', '""')
        pre = (r.preprocessed_text or "").replace('"', '""')
        output.write(f'"{raw}","{pre}","{r.sentiment or ""}","{r.aspect or ""}","{r.date or ""}"\n')
    resp = Response(output.getvalue(), mimetype="text/csv")
    resp.headers["Content-Disposition"] = f"attachment; filename=dataset_{dataset.id}.csv"
    return resp


@app.route('/export_pdf/<int:dataset_id>')
@login_required
def export_pdf(dataset_id):
    # Produces a basic HTML report (downloadable). Creating a real PDF requires extra libs (we produce HTML to download).
    dataset = Dataset.query.get_or_404(dataset_id)
    reviews = Review.query.filter_by(dataset_id=dataset.id).all()
    html = "<html><head><meta charset='utf-8'><title>Report</title></head><body>"
    html += f"<h1>Dataset Report</h1><p>Dataset: {dataset.file_name}</p>"
    html += f"<p>Total Reviews: {len(reviews)}</p><ul>"
    for r in reviews[:1000]:
        html += f"<li>{(r.raw_text or '')} — {(r.sentiment or '')}</li>"
    html += "</ul></body></html>"
    resp = Response(html, mimetype="text/html")
    resp.headers["Content-Disposition"] = f"attachment; filename=dataset_{dataset.id}_report.html"
    return resp


# ---------- Export combined CSV/PDF ----------
@app.route('/export_csv_all')
@login_required
def export_csv_all():
    if current_user.role == 'admin':
        reviews_q = Review.query.order_by(Review.id.asc()).all()
    else:
        reviews_q = Review.query.join(Dataset).filter(Dataset.user_id == current_user.id).order_by(Review.id.asc()).all()
    output = io.StringIO()
    output.write("raw_text,preprocessed_text,sentiment,aspect,date\n")
    for r in reviews_q:
        raw = (r.raw_text or "").replace('"', '""')
        pre = (r.preprocessed_text or "").replace('"', '""')
        output.write(f'"{raw}","{pre}","{r.sentiment or ""}","{r.aspect or ""}","{r.date or ""}"\n')
    resp = Response(output.getvalue(), mimetype="text/csv")
    resp.headers["Content-Disposition"] = "attachment; filename=all_reviews.csv"
    return resp


@app.route('/export_pdf_all')
@login_required
def export_pdf_all():
    if current_user.role == 'admin':
        reviews_q = Review.query.order_by(Review.id.asc()).all()
    else:
        reviews_q = Review.query.join(Dataset).filter(Dataset.user_id == current_user.id).order_by(Review.id.asc()).all()
    html = "<html><head><meta charset='utf-8'><title>Combined Report</title></head><body>"
    html += f"<h1>Combined Report</h1><p>Total Reviews: {len(reviews_q)}</p><ul>"
    for r in reviews_q[:2000]:
        html += f"<li>{(r.raw_text or '')} — {(r.sentiment or '')}</li>"
    html += "</ul></body></html>"
    resp = Response(html, mimetype="text/html")
    resp.headers["Content-Disposition"] = "attachment; filename=all_reviews_report.html"
    return resp


# ---------- CLI ----------
@app.cli.command('initdb')
def initdb_command():
    db.create_all()
    print('Initialized the database.')


# --- Compatibility alias (keep templates safe) ---
@app.route('/analyze/<int:dataset_id>')
@login_required
def analyze_dataset(dataset_id):
    return redirect(url_for('analysis', dataset_id=dataset_id))

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin123':
            user = User.query.filter_by(email='admin').first()
            if user:
                if not user.active:
                    flash('Admin account is deactivated.', 'danger')
                    return render_template('admin/login.html')
                user.last_active = datetime.utcnow()
                db.session.commit()
                login_user(user)
                create_log("Admin logged in", user.id)
                flash('Admin logged in successfully', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Admin user not found in database', 'danger')
        else:
            flash('Invalid admin credentials', 'danger')
    return render_template('admin/login.html')


if __name__ == '__main__':
    app.run(debug=True)
