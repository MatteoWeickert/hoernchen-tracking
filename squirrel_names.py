"""
Squirrel Video Matcher - Interactive Version
Input video info → Script shows which squirrels are in the video
"""

import cv2
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def load_excel_data(excel_path, study_site, box_nr):
    """Loads and filters Excel data"""
    print("\n" + "=" * 70)
    print("Loading squirrel data...")
    print("=" * 70)
    
    # Load Excel
    df = pd.read_excel(excel_path)
    print(f"✓ Excel loaded: {len(df)} rows total")
    
    # Show available columns
    print(f"  Columns: {df.columns.tolist()}")
    
    # Filter
    if 'study_site' in df.columns:
        df = df[df['study_site'] == study_site]
        print(f"✓ Filter study_site='{study_site}': {len(df)} rows")
    
    if 'box_nr' in df.columns:
        df = df[df['box_nr'] == box_nr]
        print(f"✓ Filter box_nr={box_nr}: {len(df)} rows")
    
    if len(df) == 0:
        raise ValueError("No data after filtering!")
    
    # Create datetime
    print("\n  Processing timestamps...")
    
    sample_date = df['date'].iloc[0]
    sample_time = df['time'].iloc[0]
    
    print(f"  Example date: {sample_date}")
    print(f"  Example time: {sample_time}")
    
    # Convert date to datetime
    date_col = pd.to_datetime(df['date'])
    
    # Handle time - ROBUST for mixed formats!
    from datetime import time as dt_time
    
    print("  → Analyzing time column...")
    
    # Check all types in the column
    time_types = df['time'].apply(type).unique()
    print(f"  → Found types: {[t.__name__ for t in time_types]}")
    
    # Convert each row individually
    datetime_list = []
    
    for idx, row in df.iterrows():
        date_val = pd.to_datetime(row['date'])
        time_val = row['time']
        
        if isinstance(time_val, float):
            # Excel decimal time
            time_delta = pd.Timedelta(days=time_val)
            dt = date_val + time_delta
        elif isinstance(time_val, dt_time):
            # time object
            dt = pd.Timestamp.combine(date_val.date(), time_val)
        elif isinstance(time_val, pd.Timestamp):
            # Already datetime - combine date from date_val with time from time_val
            dt = pd.Timestamp.combine(date_val.date(), time_val.time())
        else:
            # String or other - try to parse
            try:
                time_dt = pd.to_datetime(time_val).time()
                dt = pd.Timestamp.combine(date_val.date(), time_dt)
            except:
                print(f"  ⚠️  Could not parse row {idx}: time={time_val} (Type: {type(time_val)})")
                dt = date_val
        
        datetime_list.append(dt)
    
    df['datetime'] = datetime_list
    
    print(f"✓ Datetime created!")
    print(f"  Example: {df['datetime'].iloc[0]}")
    
    # Show summary
    print(f"\n✓ Squirrels in data: {df['name'].unique().tolist()}")
    print(f"✓ Time range: {df['datetime'].min().strftime('%d.%m.%Y %H:%M:%S')} to {df['datetime'].max().strftime('%d.%m.%Y %H:%M:%S')}")
    
    return df


def get_video_info(video_path):
    """Loads video properties"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    duration_sec = total_frames / fps
    duration_min = duration_sec / 60
    
    return {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'duration_sec': duration_sec,
        'duration_min': duration_min
    }


def find_squirrels_in_video(video_start_time, video_duration_sec, squirrel_data, tolerance_sec=60):
    """Finds which squirrels are in the video time window"""
    
    video_end_time = video_start_time + timedelta(seconds=video_duration_sec)
    tolerance = timedelta(seconds=tolerance_sec)
    
    print("\n" + "=" * 70)
    print("Searching for matches...")
    print("=" * 70)
    
    print(f"\nVideo time window:")
    print(f"  Start: {video_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End:   {video_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Tolerance: ±{tolerance_sec} seconds")
    
    # Find timestamps in video time window (with tolerance)
    mask = (
        (squirrel_data['datetime'] >= (video_start_time - tolerance)) &
        (squirrel_data['datetime'] <= (video_end_time + tolerance))
    )
    
    matches = squirrel_data[mask].copy()
    
    if len(matches) > 0:
        print(f"\n✓ {len(matches)} timestamps found in video!")
        
        # Sort by seconds since video start
        matches['seconds_since_start'] = (
            matches['datetime'] - video_start_time
        ).dt.total_seconds()
        
        matches = matches.sort_values('seconds_since_start')
        
        # Group by squirrel
        squirrel_counts = matches['name'].value_counts()
        
        print("\nSquirrels in video:")
        for name, count in squirrel_counts.items():
            print(f"  - {name}: {count} timestamps")
        
        # Show details
        print("\nAll timestamps:")
        print("-" * 70)
        
        for idx, row in matches.iterrows():
            time_str = row['datetime'].strftime('%H:%M:%S')
            sec_from_start = row['seconds_since_start']
            min_sec = f"{int(sec_from_start//60)}:{int(sec_from_start%60):02d}"
            
            print(f"  {time_str} ({min_sec:>6s}) - {row['name']} ({row.get('sex', 'N/A')})")
        
        return matches
    else:
        print("\n⚠️  No squirrels found in video time window!")
        print("\nPossible reasons:")
        print("  - Wrong start time?")
        print("  - Wrong date?")
        print("  - No squirrels in this time period?")
        
        # Show next available timestamps
        print("\nNext timestamps in data:")
        next_timestamps = squirrel_data.nsmallest(5, 'datetime')
        for _, row in next_timestamps.iterrows():
            print(f"  - {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')} ({row['name']})")
        
        return pd.DataFrame()


def create_annotated_video(video_path, video_start_time, matches, output_path):
    """Creates video with squirrel names"""
    
    if len(matches) == 0:
        print("\n⚠️  No matches - cannot create video")
        return
    
    print("\n" + "=" * 70)
    print("Creating annotated video...")
    print("=" * 70)
    
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create Frame → Squirrel Mapping
    frame_to_squirrel = {}
    
    for _, row in matches.iterrows():
        # Calculate frame number for this timestamp
        seconds = row['seconds_since_start']
        frame_num = int(seconds * fps)
        
        if 0 <= frame_num < total_frames:
            frame_to_squirrel[frame_num] = {
                'name': row['name'],
                'id': row.get('ID', 'N/A'),
                'sex': row.get('sex', 'N/A'),
                'time': row['datetime'].strftime('%H:%M:%S')
            }
    
    # Count visits per squirrel
    visit_counts = {}
    for name in matches['name'].unique():
        visit_counts[name] = 0
    
    # Get squirrel info for the stats box
    squirrel_info = {}
    for name in matches['name'].unique():
        first_occurrence = matches[matches['name'] == name].iloc[0]
        squirrel_info[name] = {
            'sex': first_occurrence.get('sex', 'N/A'),
            'id': first_occurrence.get('ID', 'N/A')
        }
    
    print(f"  Processing {total_frames} frames...")
    
    frame_count = 0
    current_squirrel = None
    last_squirrel_frame = -1  # Frame of last squirrel sighting
    DISAPPEAR_AFTER = int(fps * 3)  # Squirrel disappears after 3 seconds without signal
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check every 30 frames for new squirrel
        if frame_count % 30 == 0:
            # Find next match (within ±60 frames = ±2 seconds)
            found_squirrel = None
            min_distance = float('inf')
            
            for match_frame in frame_to_squirrel.keys():
                distance = abs(match_frame - frame_count)
                if distance < min_distance and distance <= 60:
                    min_distance = distance
                    found_squirrel = frame_to_squirrel[match_frame]
            
            if found_squirrel:
                # New squirrel found
                if current_squirrel is None or current_squirrel['name'] != found_squirrel['name']:
                    # New squirrel or different squirrel → Count visit
                    visit_counts[found_squirrel['name']] += 1
                
                current_squirrel = found_squirrel
                last_squirrel_frame = frame_count
        
        # Squirrel disappears after DISAPPEAR_AFTER frames without new match
        if current_squirrel and (frame_count - last_squirrel_frame) > DISAPPEAR_AFTER:
            current_squirrel = None
        
        # === STATISTICS BOX (bottom right) - ALWAYS VISIBLE ===
        stats_height = 60 + len(squirrel_info) * 35
        stats_width = 300
        stats_x = width - stats_width - 10
        stats_y = height - stats_height - 10
        
        # Black background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (stats_x, stats_y), (width - 10, height - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Header
        cv2.putText(frame, "Visit Statistics", 
                   (stats_x + 10, stats_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2)
        
        # Draw line under header
        cv2.line(frame, (stats_x + 10, stats_y + 35), (width - 20, stats_y + 35), (100, 100, 100), 1)
        
        # List all squirrels with their visit counts
        y_offset = stats_y + 55
        for name in sorted(squirrel_info.keys()):
            info = squirrel_info[name]
            sex_symbol = "♀" if info['sex'] == 'female' else "♂" if info['sex'] == 'male' else "?"
            visits = visit_counts[name]
            
            # Highlight current squirrel
            if current_squirrel and current_squirrel['name'] == name:
                color = (0, 255, 255)  # Cyan for active squirrel
                thickness = 2
            else:
                color = (200, 200, 200)  # Gray for inactive
                thickness = 1
            
            text = f"{name} {sex_symbol}: {visits} visits"
            cv2.putText(frame, text, 
                       (stats_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            y_offset += 35
        
        # === CURRENT SQUIRREL ANNOTATION (top left) - only when active ===
        if current_squirrel:
            # Black background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (550, 160), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Current time
            current_time = video_start_time + timedelta(seconds=frame_count/fps)
            
            # Text
            y = 40
            cv2.putText(frame, f"Squirrel: {current_squirrel['name']}", 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"ID: {current_squirrel['id']} ({current_squirrel['sex']})", 
                       (20, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {current_time.strftime('%H:%M:%S')}", 
                       (20, y+75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Visit #{visit_counts[current_squirrel['name']]}", 
                       (20, y+110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        
        out.write(frame)
        frame_count += 1
        
        # Progress
        if frame_count % 300 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"    {progress:.1f}% done...")
    
    cap.release()
    out.release()
    
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\n✓ Video complete!")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Path: {output_path}")
    
    # Show visit statistics
    print(f"\n📊 Visit Statistics:")
    for name, count in sorted(visit_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {count} visits")


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    print("=" * 70)
    print("SQUIRREL VIDEO MATCHER - Interactive Version")
    print("=" * 70)
    
    # CONFIGURATION - ADJUST HERE!
    print("\n📁 FILE PATHS:")
    VIDEO_PATH = r"data\TrepN_08_10.mp4"
    EXCEL_PATH = r"data\antenna_master_sheet.xlsx"
    print(f"  Video: {VIDEO_PATH}")
    print(f"  Excel: {EXCEL_PATH}")
    
    print("\n📍 VIDEO INFORMATION:")
    STUDY_SITE = "trep_n"
    BOX_NR = 4
    VIDEO_START_TIME = datetime(2024, 10, 8, 9, 10, 19)  # Year, Month, Day, Hour, Minute, Second
    
    print(f"  Study Site: {STUDY_SITE}")
    print(f"  Box Nr: {BOX_NR}")
    print(f"  Start Time: {VIDEO_START_TIME.strftime('%d.%m.%Y %H:%M:%S')}")
    
    TIME_TOLERANCE = 60  # Seconds
    print(f"  Tolerance: ±{TIME_TOLERANCE} seconds")
    
    # Output
    OUTPUT_CSV = "squirrels_in_video.csv"
    OUTPUT_VIDEO = r"output\annotated_video.mp4"
    
    try:
        # 1. Load video properties
        print("\n" + "=" * 70)
        print("Loading video information...")
        print("=" * 70)
        
        video_info = get_video_info(VIDEO_PATH)
        print(f"✓ Video: {video_info['width']}x{video_info['height']}")
        print(f"✓ FPS: {video_info['fps']}")
        print(f"✓ Duration: {video_info['duration_min']:.2f} minutes ({video_info['duration_sec']:.1f} seconds)")
        print(f"✓ Frames: {video_info['total_frames']}")
        
        # 2. Load Excel data
        squirrel_data = load_excel_data(EXCEL_PATH, STUDY_SITE, BOX_NR)
        
        # 3. Find matches
        matches = find_squirrels_in_video(
            video_start_time=VIDEO_START_TIME,
            video_duration_sec=video_info['duration_sec'],
            squirrel_data=squirrel_data,
            tolerance_sec=TIME_TOLERANCE
        )
        
        # 4. Save
        if len(matches) > 0:
            # Save CSV
            matches_export = matches[['datetime', 'name', 'ID', 'sex', 'seconds_since_start']].copy()
            matches_export.to_csv(OUTPUT_CSV, index=False)
            print(f"\n✓ List saved: {OUTPUT_CSV}")
            
            # Create video?
            print("\n" + "=" * 70)
            create_video = input("Create annotated video? (y/n): ").strip().lower()
            
            if create_video == 'y':
                Path(OUTPUT_VIDEO).parent.mkdir(exist_ok=True)
                create_annotated_video(VIDEO_PATH, VIDEO_START_TIME, matches, OUTPUT_VIDEO)
            else:
                print("  Video creation skipped")
        
        # DONE
        print("\n" + "=" * 70)
        print("DONE! 🎉")
        print("=" * 70)
        
        if len(matches) > 0:
            print(f"\n✓ {len(matches)} squirrel timestamps found")
            print(f"✓ CSV: {OUTPUT_CSV}")
            if Path(OUTPUT_VIDEO).exists():
                print(f"✓ Video: {OUTPUT_VIDEO}")
        else:
            print("\n⚠️  No squirrels in video time window")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()