from flask import Flask, request, jsonify, session, redirect, url_for, render_template
import pandas as pd
import os
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='static')
app.secret_key = 'b4181cbfc55ef8a45297f5d5b1104921ebc79ceedfbfcdf3a6cd742fa1c8519e'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Initialize in-memory data structures
users = {
    "admin": {"password": "admin123", "role": "admin"},
    "user1": {"password": "user123", "role": "user"},
    "tjrourk": {"password": "passw0rd", "role": "admin"}
}
work_queues = {}
task_locks = {}
completed_records = {} 

# Define five standard spreadsheet formats
STANDARD_FORMATS = [
    {"Record ID", "Company Name", "Data Field", "Value"},
    {"ID", "Name", "Category", "Amount"},
    {"Unique ID", "Client", "Type", "Details"},
    {"A_ID","B_ID","A_NAME","B_NAME","A_MAILADDRESS1","B_MAILADDRESS1"},
    {"CD Number","Company Name","Country Name","DUNS"}
]

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to validate file format
def validate_format(df):
    uploaded_columns = set(df.columns)
    for format_set in STANDARD_FORMATS:
        if format_set.issubset(uploaded_columns):
            return True
    return False

# Utility functions
def load_csv_to_queue(file_path, queue_name):
    df = pd.read_csv(file_path)
    if not validate_format(df):
        return False, 'Invalid file format. File does not match any predefined standard formats.'
    # Initialize missing columns for locking mechanism
    df['locked_by'] = None
    df['lock_timestamp'] = None
    df['Status'] = 'Open'
    df['Assigned To'] = None
    df['Last Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df['Created Time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if queue_name in work_queues:
        work_queues[queue_name] = pd.concat([work_queues[queue_name], df], ignore_index=True)
    else:
        work_queues[queue_name] = df
    return True, 'File uploaded successfully'

def get_dashboard_metrics(queue_name):
    if queue_name not in work_queues:
        return {}

    df = work_queues[queue_name]
    metrics = {
        "backlog": len(df[df['Status'] == 'Open']),
        "added_last_24h": len(df[df['Last Updated'] >= (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')]),
        "completed_last_24h": len(df[df['Status'] == 'Completed']),
        "total_completed": len(df[df['Status'] == 'Completed']),
        "average_throughput": len(df[df['Status'] == 'Completed']) / ((datetime.now() - datetime.strptime(df['Last Updated'].min(), '%Y-%m-%d %H:%M:%S')).total_seconds() / 3600)
    }
    return metrics

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    #print("Rendering home.html")
    return render_template('home.html', queues=work_queues.keys())

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            session['username'] = username
            session['role'] = users[username]['role']
            return redirect(url_for('index'))
        else:
            return "Invalid credentials", 401
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'role' not in session or session['role'] != 'admin':
        return "Unauthorized", 403

    if request.method == 'POST':
        queue_name = request.form['queue_name']
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            success, message = load_csv_to_queue(file_path, queue_name)
            if success:
                return redirect(url_for('index'))
            else:
                return jsonify({'error': message}), 400
    return render_template('upload.html', admin=True)

@app.route('/queues')
def queues():
    if not work_queues:
        return "No queues available", 404
    return render_template('queues.html', queues=work_queues.keys())

@app.route('/queue/<queue_name>', methods=['GET', 'POST'])
def queue(queue_name):
    if queue_name not in work_queues:
        return "Queue not found", 404
    
    df = work_queues[queue_name]
    if request.method == 'POST':
        search_type = request.form['search_type']
        search_query = request.form['search_query']
        if search_type and search_query:
            if search_type in df.columns:
                df[search_type] = df[search_type].astype(str)  # Convert to string
                df = df[df[search_type].str.contains(search_query, case=False, na=False)]
            else:
                return f"Error: Column {search_type} does not exist in the DataFrame.", 400

    metrics = get_dashboard_metrics(queue_name)
    return render_template('queue.html', queue_name=queue_name, metrics=metrics, tasks=df.to_dict(orient='records'))

@app.route('/queue/<queue_name>/get_next')
def get_next_task(queue_name):
    if queue_name not in work_queues:
        return "Queue not found", 404

    df = work_queues[queue_name]
    # Filter for tasks that are open and not currently assigned
    next_task = df[(df['Status'] == 'Open') & (df['Assigned To'].isnull())].head(1)
    if not next_task.empty:
        task_id = next_task.index[0]
        # Optionally lock the task or update its status to 'In Progress'
        df.at[task_id, 'Status'] = 'In Progress'
        return redirect(url_for('task', queue_name=queue_name, task_id=task_id))
    else:
        return "No available tasks", 404

@app.route('/queue/<queue_name>/task/<int:task_id>', methods=['GET', 'POST'])
def task(queue_name, task_id):
    if queue_name not in work_queues:
        return "Queue not found", 404

    df = work_queues.get(queue_name, pd.DataFrame())
    if task_id < 0 or task_id >= len(df):
        return "Task not found", 404
    
    
    #task = df.loc[task_id]

    if request.method == 'POST':
        status = request.form['status']
        result = request.form.get('result', '')
        df.loc[task_id, 'Status'] = status
        df.loc[task_id, 'Assigned To'] = session['username']
        df.loc[task_id, 'Last Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if result:
            df.loc[task_id, 'Result'] = result
        for key in request.form:
            if key in df.columns and key != 'Status' and key != 'Result':  # Prevent repetitive status/result update
                df.loc[task_id, key] = request.form[key]
        
        # Update status and result separately
        df.loc[task_id, 'Status'] = request.form.get('status', 'Open')
        df.loc[task_id, 'Result'] = request.form.get('result', '')

        # Lock management
        df.loc[task_id, 'locked_by'] = None
        df.loc[task_id, 'lock_timestamp'] = None

        work_queues[queue_name] = df  # Ensure the main data structure is updated
        return redirect(url_for('queue', queue_name=queue_name))
    
    # Check if someone else holds a valid (unexpired) lock
    now = datetime.now()
    lock_owner = df.at[task_id, 'locked_by']
    lock_timestamp = pd.to_datetime(df.at[task_id, 'lock_timestamp'], errors='coerce')
    
    # Convert to datetime if not NaN
    if pd.notna(lock_timestamp):
        lock_timestamp = pd.to_datetime(lock_timestamp, errors='coerce')
    
    lock_expired = False
    if pd.notna(lock_timestamp):
        if now - lock_timestamp > timedelta(hours=1):
            lock_expired = True
    
    if pd.isna(lock_owner) or lock_owner == '' or lock_expired:
        # Lock is free or expired -> Acquire lock
        df.at[task_id, 'locked_by'] = session['username']
        df.at[task_id, 'lock_timestamp'] = now
        work_queues[queue_name] = df
    else:
        # Lock is taken by someone else
        if lock_owner != session['username']:
            # Another user has this locked, and it's not expired
            return f"Task {task_id} is locked by {lock_owner}. Please try again later.", 403

    task_data = df.iloc[task_id].to_dict()
    next_task_id = task_id + 1 if task_id + 1 < len(df) else None
    
    
    
    return render_template('task.html', task=task_data, queue_name=queue_name, task_id=task_id, next_task_id=next_task_id)

@app.route('/queue/<queue_name>/task/<int:task_id>/update', methods=['POST'])
def update_task(queue_name, task_id):
    if queue_name not in work_queues:
        return "Queue not found", 404

    df = work_queues[queue_name]
    if task_id < 0 or task_id >= len(df):
        return "Task not found", 404

    # Process the form data and directly update the DataFrame
    new_status = request.form.get('status', 'open')
    # Set the status based on the action specified in the form
    if new_status == "Completed":
        # Copy the row
        completed_row = df.loc[[task_id]].copy()
        # Set the Completion Time if not already set
        completed_row['Status'] = 'Completed'
        completed_row['Completion Time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # If assigned user is needed
        completed_row['Assigned To'] = session.get('username')

        # Initialize completed_records[queue_name] if needed
        if queue_name not in completed_records:
            completed_records[queue_name] = completed_row
        else:
            completed_records[queue_name] = pd.concat([
                completed_records[queue_name],
                completed_row
            ], ignore_index=True)

        # Remove the row from active queue
        df.drop(task_id, inplace=True)
        df.reset_index(drop=True, inplace=True)
        work_queues[queue_name] = df
    elif new_status == "Not Found":
        df.loc[task_id, 'Status'] = 'Not Found'
    elif new_status == "Skip":
        df.loc[task_id, 'Status'] = 'Open'
        df.loc[task_id, 'Assigned To'] = None  # Clear the assigned user if skipped
    elif new_status == "Duplicate":
        # If handling duplicates, ensure you add logic here for how to process them
        pass
    else:
        # If user picks In Progress, or something else
        df.loc[task_id, 'Status'] = action
        df.loc[task_id, 'Last Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


    # Update other fields from the form
    for key in request.form:
        if key in df.columns and key != 'status':  # Skip 'status' as it's already handled
            df.loc[task_id, key] = request.form[key]

    # Ensure the main data structure is updated
    work_queues[queue_name] = df  
    
    df.to_csv(f'{queue_name}.csv')
    return redirect(url_for('queue', queue_name=queue_name))


def find_duplicates(df):
    # Assuming 'Company Name' and 'Email' are the criteria for finding duplicates
    duplicates = df.duplicated(subset=['Company Name', 'Email'], keep=False)
    return df[duplicates]

@app.route('/queue/<queue_name>/duplicates')
def show_duplicates(queue_name):
    if queue_name not in work_queues:
        return "Queue not found", 404

    df = work_queues[queue_name]
    duplicate_df = find_duplicates(df)
    if duplicate_df.empty:
        return "No duplicates found", 404

    # Convert DataFrame to a suitable format for HTML rendering
    duplicates = duplicate_df.to_dict(orient='records')
    return render_template('duplicates.html', queue_name=queue_name, duplicates=duplicates)

@app.route('/queue/<queue_name>/resolve_duplicates', methods=['POST'])
def resolve_duplicates(queue_name):
    survivor_id = request.form['survivor']
    df = work_queues[queue_name]

    # Mark all as duplicates except the survivor
    df['Status'] = df.apply(lambda x: 'Duplicate' if str(x['Record ID']) != survivor_id else 'Active', axis=1)

    # Update the DataFrame in your queue
    work_queues[queue_name] = df
    return redirect(url_for('queue', queue_name=queue_name))

@app.route('/admin/unlock/<queue_name>/<int:task_id>', methods=['POST'])
def admin_unlock(queue_name, task_id):
    # Only allow if user is an admin
    if 'role' not in session or session['role'] != 'admin':
        return "Unauthorized", 403

    # Check the queue and task
    if queue_name not in work_queues:
        return "Queue not found", 404

    df = work_queues[queue_name]
    if task_id < 0 or task_id >= len(df):
        return "Task not found", 404

    # Manually clear the lock fields
    df.at[task_id, 'locked_by'] = None
    df.at[task_id, 'lock_timestamp'] = None

    # Update the in-memory queue
    work_queues[queue_name] = df

    return redirect(url_for('queue', queue_name=queue_name))

def user_is_admin():
    return session.get("role") == "admin"

@app.route('/admin/dashboard')
def admin_dashboard():
    # Restrict access to admins
    if session.get('role') != 'admin':
        return "Unauthorized", 403

    now = datetime.now()
    last_24h = now - timedelta(hours=24)

    # Backlog from active queues
    total_backlog = 0
    for qname, df in work_queues.items():
        total_backlog += len(df[df['Status'] == 'Open'])

    # Completed stats from completed_records
    all_completed_list = []
    for qname, cdf in completed_records.items():
        # Convert your time columns to datetime so we can filter
        if 'Completion Time' in cdf.columns:
            cdf['Completion Time'] = pd.to_datetime(cdf['Completion Time'], errors='coerce')
        if 'Created Time' in cdf.columns:
            cdf['Created Time'] = pd.to_datetime(cdf['Created Time'], errors='coerce')

        # Mark which queue it came from (optional)
        cdf['Queue Name'] = qname
        all_completed_list.append(cdf)

    # Combine all completed data
    if all_completed_list:
        all_completed = pd.concat(all_completed_list, ignore_index=True)
    else:
        all_completed = pd.DataFrame()

    total_completed_24h = 0
    avg_time_spent_hours = 0
    completion_by_analyst = {}
    throughput_by_analyst = {}

    if not all_completed.empty:
        # Completed in Last 24 hours
        completed_24h_df = all_completed[all_completed['Completion Time'] >= last_24h]
        total_completed_24h = len(completed_24h_df)

        # Average Time Spent
        # If you store Created Time, compute difference
        if 'Created Time' in all_completed.columns:
            all_completed['Time Spent'] = all_completed['Completion Time'] - all_completed['Created Time']
            avg_time_spent = all_completed['Time Spent'].mean()
            if pd.notnull(avg_time_spent):
                avg_time_spent_hours = avg_time_spent.total_seconds() / 3600

        # Completion count by analyst
        if 'Assigned To' in all_completed.columns:
            completion_count_series = all_completed.groupby('Assigned To')['Status'].count()
            completion_by_analyst = completion_count_series.to_dict()

        # Throughput by analyst
        if 'Assigned To' in all_completed.columns and 'Last Updated' in all_completed.columns:
            all_completed['Last Updated'] = pd.to_datetime(all_completed['Last Updated'], errors='coerce')
            all_completed['Time In Progress'] = all_completed['Completion Time'] - all_completed['Last Updated']
            all_completed['Hours'] = all_completed['Time In Progress'].dt.total_seconds() / 3600
            grp = all_completed.groupby('Assigned To').agg({'Status': 'count', 'Hours': 'sum'})
            grp['Throughput (records/hour)'] = grp['Status'] / grp['Hours']
            grp['Throughput (records/hour)'].fillna(0, inplace=True)
            throughput_by_analyst = grp['Throughput (records/hour)'].to_dict()

    # Render a template, passing these computed values
    return render_template('admin_dashboard.html',
                           total_backlog=total_backlog,
                           total_completed_24h=total_completed_24h,
                           avg_time_spent_hours=avg_time_spent_hours,
                           completion_by_analyst=completion_by_analyst,
                           throughput_by_analyst=throughput_by_analyst)

def create_queue():
    if not user_is_admin():
        return "Unauthorized", 403

    if request.method == 'POST':
        new_queue_name = request.form['queue_name']
        if new_queue_name in work_queues:
            return "Queue already exists", 400
        # Create an empty DataFrame with some columns
        work_queues[new_queue_name] = pd.DataFrame(
            columns=["Record ID", "Status", "Assigned To", "Last Updated", "Created Time", "Completion Time", "locked_by", "lock_timestamp"]
        )
        return redirect(url_for('index'))
    
    return '''
    <form method="POST">
      <label>Queue Name:</label>
      <input type="text" name="queue_name" />
      <button type="submit">Create</button>
    </form>
    '''

@app.route('/admin/delete_queue/<queue_name>', methods=['POST'])
def delete_queue(queue_name):
    if not user_is_admin():
        return "Unauthorized", 403

    if queue_name in work_queues:
        del work_queues[queue_name]
        return redirect(url_for('index'))
    else:
        return "Queue not found", 404

def unlock_expired_locks():
    now = datetime.now()
    for queue_name, df in work_queues.items():
        if 'lock_timestamp' not in df.columns:
            continue
        expired_mask = (
            df['lock_timestamp'].notna() &
            ((pd.to_datetime(df['lock_timestamp'], errors='coerce') + timedelta(hours=1)) < now)
        )
        if any(expired_mask):
            df.loc[expired_mask, ['locked_by', 'lock_timestamp']] = [None, None]
            work_queues[queue_name] = df

@app.before_request
def check_expired_locks():
    unlock_expired_locks()




if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)

