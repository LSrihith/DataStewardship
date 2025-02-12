from flask import Flask, request, jsonify, session, redirect, url_for, render_template
import pandas as pd
import os
from datetime import datetime, timedelta
from collections import defaultdict
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, static_folder='static')
app.secret_key = 'b4181cbfc55ef8a45297f5d5b1104921ebc79ceedfbfcdf3a6cd742fa1c8519e'
UPLOAD_FOLDER = 'uploads'
ACTIVE_QUEUES_FOLDER = 'active_queues'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
os.makedirs(ACTIVE_QUEUES_FOLDER, exist_ok=True)

USER_FILE = 'users.csv'
work_queues = {}
task_locks = {}
completed_records = {} 

# A dictionary: queue_permissions[queue_name][username] = True/False
# If user is admin, they can always access. Otherwise, check this dict.
queue_permissions = defaultdict(lambda: defaultdict(bool))



# Define five standard spreadsheet formats
STANDARD_FORMATS = [
    {"Record ID", "Company Name", "Data Field", "Value"},
    {"ID", "Name", "Category", "Amount"},
    {"Unique ID", "Client", "Type", "Details"},
    {"A_ID","B_ID","A_NAME","B_NAME","A_MAILADDRESS1","B_MAILADDRESS1"},
    {"CD Number","Company Name","Country Name","DUNS"}
]

# Load the existing users.csv
if os.path.exists(USER_FILE):
    df = pd.DataFrame(columns=["username", "password", "role"])
    df = pd.read_csv(USER_FILE, dtype=str)

    # There are already 3 users in the csv file. This function hashs the password. 
    """for index, row in df.iterrows():
        password = row["password"]
        if not password.startswith("pbkdf2:sha256"):  # Ensure it's not already hashed
            df.at[index, "password"] = generate_password_hash(password)"""

    # Save the updated CSV with hashed passwords
    df.to_csv(USER_FILE, index=False)

# Load users from CSV
def load_users():
    # Load users from CSV into a dictionary.
    if os.path.exists(USER_FILE):
        df = pd.read_csv(USER_FILE, dtype=str)
        if "username" not in df.columns or "password" not in df.columns or "role" not in df.columns:
            raise ValueError("CSV file is missing required columns: username, password, role")
        return df.set_index("username").T.to_dict()
    return {}

# Save users back to CSV
def save_users(users):
    # Save users dictionary back to CSV.
    df = pd.DataFrame.from_dict(users, orient='index')  # Convert dictionary to DataFrame
    df.reset_index(inplace=True)  # Convert index (username) back to a column
    df.rename(columns={"index": "username"}, inplace=True)  # Ensure column name is correct
    df.to_csv(USER_FILE, index=False)  # Save without an index column

# Verify password against the hashed password
def verify_password(hashed_password, provided_password):
    # Check if the provided password matches the stored hash.
    return check_password_hash(hashed_password, provided_password)

@app.route('/login', methods=['GET', 'POST'])
def login():
    users = load_users() # load users from csv
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and verify_password(users[username]['password'], password):
            session['username'] = username
            session['role'] = users[username]['role']
            return redirect(url_for('index'))
        else:
            return "Invalid credentials", 401
    return render_template('login.html')

@app.route('/admin/manage_users', methods=['GET', 'POST'])
def manage_users():
    # Admin page to manage users (create users and update passwords).
    if session.get('role') != 'admin':
        return "Unauthorized", 403

    users = load_users()
    message = ""

    if request.method == 'POST':
        action = request.form['action']
        username = request.form['username']

        if action == "create":
            if username in users:
                message = "User already exists!"
            else:
                password = request.form['password']
                role = request.form['role']
                users[username] = {
                    "username": username,
                    "password": generate_password_hash(password),
                    "role": role
                }
                save_users(users)
                message = "User created successfully!"

        elif action == "update_password":
            if username not in users:
                message = "User not found!"
            else:
                current_password = request.form['current_password']
                new_password = request.form['new_password']

                # Verify the current password
                if verify_password(users[username]['password'], current_password):
                    users[username]['password'] = generate_password_hash(new_password)
                    save_users(users)
                    message = "Password updated successfully!"
                else:
                    message = "Incorrect current password!"

    return render_template('manage_users.html', users=users, message=message)

@app.route('/update_password', methods=['GET', 'POST'])
def update_password():
    # Allow logged-in users to change their password.
    if 'username' not in session:
        return redirect(url_for('login'))  # Redirect if not logged in

    users = load_users()
    username = session['username']  # Get logged-in user
    message = ""

    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']

        # Verify the current password
        if username in users and verify_password(users[username]['password'], current_password):
            users[username]['password'] = generate_password_hash(new_password)  # Update password
            save_users(users)  # Save changes
            message = "Password updated successfully!"
        else:
            message = "Incorrect current password!"

    return render_template('update_password.html', message=message)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

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

def user_is_admin():
    return session.get("role") == "admin"

def can_user_access_queue(username, queue_name):
    # Admins have universal access; otherwise check queue_permissions.
    if user_is_admin():
        return True
    return queue_permissions[queue_name].get(username, False)

# Utility functions
def load_csv_to_queue(file_path, queue_name):
    df = pd.read_csv(file_path)
    if not validate_format(df):
        return False, 'Invalid file format. File does not match any predefined standard formats.'
    
    # If the CSV has "Status" column, drop rows that are already "Completed"
    if 'Status' in df.columns:
        df = df[df['Status'] != 'Completed'].copy()
    
    # Initialize missing columns
    if 'locked_by' not in df.columns:
        df['locked_by'] = None
    if 'lock_timestamp' not in df.columns:
        df['lock_timestamp'] = None
    if 'Status' not in df.columns:
        df['Status'] = 'Open'
    else:
        # For any row with blank status, default to 'Open'
        df['Status'] = df['Status'].fillna('Open')
    
    if 'Assigned To' not in df.columns:
        df['Assigned To'] = None
    if 'Last Updated' not in df.columns:
        df['Last Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if 'Created Time' not in df.columns:
        df['Created Time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Combine with in-memory queue
    if queue_name in work_queues:
        combined_df = pd.concat([work_queues[queue_name], df], ignore_index=True)
        work_queues[queue_name] = combined_df
    else:
        work_queues[queue_name] = df
    
    # Save the active queue to ACTIVE_QUEUES_FOLDER
    file_out = os.path.join(ACTIVE_QUEUES_FOLDER, f'{queue_name}.csv')
    work_queues[queue_name].to_csv(file_out, index=False)
    
    return True, 'File uploaded successfully'

def load_all_queues():
    """
    Loads active queue CSV files from the ACTIVE_QUEUES_FOLDER,
    filtering out any rows with Status == 'Completed' or whose Record ID
    is already present in the corresponding completed_records CSV file.
    This ensures that after a server restart, any tasks that have been
    completed (even if not flagged in the active CSV) do not appear.
    """
    for filename in os.listdir(ACTIVE_QUEUES_FOLDER):
        if filename.endswith('.csv'):
            queue_name = filename.rsplit('.', 1)[0]
            file_path = os.path.join(ACTIVE_QUEUES_FOLDER, filename)
            try:
                # Load the CSV file into a DataFrame.
                df = pd.read_csv(file_path, dtype=str)
                # If the "Status" column exists, filter out completed rows.
                if 'Status' in df.columns:
                    df = df[df['Status'] != 'Completed'].copy()
                work_queues[queue_name] = df
                print(f"Loaded queue: {queue_name} with {len(df)} active records")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

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
    
    # Check if the current user has access to this queue
    if not can_user_access_queue(session.get('username'), queue_name):
        return "Unauthorized", 403
    
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
    df_display = df[df['Status'] != 'Completed'].copy()
    metrics = get_dashboard_metrics(queue_name)
    return render_template('queue.html', queue_name=queue_name, metrics=metrics, tasks=df_display.to_dict(orient='records'))

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
        df.to_csv(os.path.join(ACTIVE_QUEUES_FOLDER, f'{queue_name}.csv'), index=False)
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
    action = request.form.get('status', 'open')

    # Set the status based on the action specified in the form    
    if action == "Not Found":
        df.loc[task_id, 'Status'] = 'Not Found'
    elif action == "Skip":
        df.loc[task_id, 'Status'] = 'Open'
        df.loc[task_id, 'Assigned To'] = None
    else:
        df.loc[task_id, 'Status'] = action
        df.loc[task_id, 'Last Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Update other form fields
    for key in request.form:
        if key in df.columns and key != 'status':
            df.loc[task_id, key] = request.form[key]
    
    if action == "Completed":
        now = datetime.now()
        df.loc[task_id, 'Status'] = 'Completed'
        df.loc[task_id, 'Completion Time'] = now.strftime('%Y-%m-%d %H:%M:%S')
        # Append the completed row to a CSV for history
        if not os.path.exists('completed_records'):
            os.makedirs('completed_records')
        completed_file_path = os.path.join('completed_records', f'{queue_name}_completed_records.csv')
        # Extract just this one row
        completed_task = df.loc[[task_id]].copy()
        completed_task.to_csv(
            completed_file_path,
            mode='a',
            header=not os.path.exists(completed_file_path),
            index=False
        )

    df.to_csv(f'{queue_name}.csv', index=False)  # If you want to persist active queue as well
    df.to_csv(os.path.join(ACTIVE_QUEUES_FOLDER, f'{queue_name}.csv'), index=False)
    # Update the in-memory DataFrame
    work_queues[queue_name] = df
    # Finally, redirect to the queue page (the completed task is now gone)
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

@app.route('/admin/dashboard')
def admin_dashboard():
    # Restrict access to admins
    if session.get('role') != 'admin':
        return "Unauthorized", 403

    now = datetime.now()
    last_24h = now - timedelta(hours=24)

    # Calculate backlog from active queues
    total_backlog = 0
    for qname, df in work_queues.items():
        # Count tasks still marked "Open"
        total_backlog += len(df[df['Status'] == 'Open'])

    # Gather all completed records from CSV files in /completed_records
    completed_dir = 'completed_records'
    all_completed_list = []

    if os.path.exists(completed_dir):
        # For each CSV file in /completed_records, read and combine
        for filename in os.listdir(completed_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(completed_dir, filename)
                
                # Read the CSV into a DataFrame
                cdf = pd.read_csv(file_path, dtype=str)  # dtype=str helps avoid surprises
                
                # Convert relevant time columns to datetime if they exist
                if 'Completion Time' in cdf.columns:
                    cdf['Completion Time'] = pd.to_datetime(cdf['Completion Time'], errors='coerce')
                if 'Created Time' in cdf.columns:
                    cdf['Created Time'] = pd.to_datetime(cdf['Created Time'], errors='coerce')
                if 'Last Updated' in cdf.columns:
                    cdf['Last Updated'] = pd.to_datetime(cdf['Last Updated'], errors='coerce')
                
                # Optionally, parse queue name from the filename or store in the CSV
                # cdf['Queue Name'] = filename.replace('_completed_records.csv','')
                
                all_completed_list.append(cdf)

    # Combine all completed data
    if all_completed_list:
        all_completed = pd.concat(all_completed_list, ignore_index=True)
    else:
        # If no completed CSV files or they're empty
        all_completed = pd.DataFrame()

    # Compute your metrics
    total_completed_24h = 0
    avg_time_spent_hours = 0
    completion_by_analyst = {}
    throughput_by_analyst = {}

    if not all_completed.empty and 'Completion Time' in all_completed.columns:
        # Completed in last 24 hours
        completed_24h_df = all_completed[all_completed['Completion Time'] >= last_24h]
        total_completed_24h = len(completed_24h_df)

        # Average Time Spent (needs both Created Time and Completion Time)
        if 'Created Time' in all_completed.columns:
            # Subtract Created from Completion
            all_completed['Time Spent'] = all_completed['Completion Time'] - all_completed['Created Time']
            avg_time_spent = all_completed['Time Spent'].mean()
            if pd.notnull(avg_time_spent):
                avg_time_spent_hours = avg_time_spent.total_seconds() / 3600

        # Completion count by analyst
        if 'Assigned To' in all_completed.columns:
            completion_count_series = all_completed.groupby('Assigned To')['Status'].count()
            completion_by_analyst = completion_count_series.to_dict()

        # Throughput by analyst (requires Last Updated and Completion Time)
        if 'Assigned To' in all_completed.columns and 'Last Updated' in all_completed.columns:
            # Calculate how long it was "In Progress"
            all_completed['Time In Progress'] = all_completed['Completion Time'] - all_completed['Last Updated']
            all_completed['Hours'] = all_completed['Time In Progress'].dt.total_seconds() / 3600

            grp = all_completed.groupby('Assigned To').agg({'Status': 'count', 'Hours': 'sum'})
            grp['Throughput (records/hour)'] = grp['Status'] / grp['Hours']
            grp['Throughput (records/hour)'].fillna(0, inplace=True)
            throughput_by_analyst = grp['Throughput (records/hour)'].to_dict()

    # Render the dashboard template with these values
    return render_template(
        'admin_dashboard.html',
        total_backlog=total_backlog,
        total_completed_24h=total_completed_24h,
        avg_time_spent_hours=avg_time_spent_hours,
        completion_by_analyst=completion_by_analyst,
        throughput_by_analyst=throughput_by_analyst
    )


@app.route('/admin/create_queue', methods=['GET', 'POST'])
def create_queue():
    if not user_is_admin():
        return "Unauthorized", 403

    if request.method == 'POST':
        # Retrieve the fields from the form
        cd_number = request.form.get('cd_number')
        company_name = request.form.get('company_name')
        country_name = request.form.get('country_name')
        duns = request.form.get('duns')  # Optional field

        # Validate required fields
        if not (cd_number and company_name and country_name):
            return "CD Number, Company Name, and Country Name are required.", 400

        # Auto-generate a new queue name (e.g., "Queue 1", "Queue 2", etc.)
        new_queue_name = f"Queue {len(work_queues) + 1}"
        if new_queue_name in work_queues:
            new_queue_name += f"_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Create an empty DataFrame for tasks in the new queue with expected columns.
        work_queues[new_queue_name] = pd.DataFrame(
            columns=["Record ID", "Status", "Assigned To", "Last Updated",
                     "Created Time", "Completion Time", "locked_by", "lock_timestamp"]
        )

        # Create a metadata dictionary for this new queue.
        metadata = {
            "Queue Name": new_queue_name,
            "CD Number": cd_number,
            "Company Name": company_name,
            "Country Name": country_name,
            "DUNS": duns or ""  # If DUNS is not provided, store an empty string.
        }

        # Save metadata to a single CSV file that stores all queue metadata.
        metadata_file = "queue_metadata.csv"
        if os.path.exists(metadata_file):
            # Read existing metadata
            meta_df = pd.read_csv(metadata_file, dtype=str)
            # Append the new metadata row
            meta_df = meta_df.append(metadata, ignore_index=True)
        else:
            meta_df = pd.DataFrame([metadata])
        meta_df.to_csv(metadata_file, index=False)

        return redirect(url_for('index'))

    return render_template('create_queue.html')

@app.route('/admin/delete_queue/<queue_name>', methods=['POST'])
def delete_queue(queue_name):
    if not user_is_admin():
        return "Unauthorized", 403

    if queue_name in work_queues:
        del work_queues[queue_name]
        completed_file = os.path.join('completed_records', f'{queue_name}_completed_records.csv')
        if os.path.exists(completed_file):
            os.remove(completed_file)
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

@app.route('/admin/assign_permissions/<queue_name>', methods=['GET','POST'])
def assign_permissions(queue_name):
    # Allows an admin to assign which users can access a specific queue.
    users = load_users()
    if not user_is_admin():
        return "Unauthorized", 403

    # Make sure queue exists
    if queue_name not in work_queues:
        return "Queue not found", 404

    if request.method == 'POST':
        # For each user in your 'users' dict, check if the form had it "on"
        for username in users:
            can_access = request.form.get(username)  # 'on' if checked, else None
            if can_access == 'on':
                queue_permissions[queue_name][username] = True
            else:
                queue_permissions[queue_name][username] = False
        return redirect(url_for('queue', queue_name=queue_name))

    # GET: show a form with checkboxes for each user
    html = f"<h1>Assign Permissions for queue: {queue_name}</h1>"
    html += "<form method='POST'>"
    for username in users:
        has_access = queue_permissions[queue_name].get(username, False)
        checked = "checked" if has_access else ""
        html += f"<label>{username}</label> "
        html += f"<input type='checkbox' name='{username}' {checked}><br>"
    html += "<button type='submit'>Save</button></form>"
    return html

@app.before_request
def check_expired_locks():
    unlock_expired_locks()

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    load_all_queues()
    app.run(debug=True)