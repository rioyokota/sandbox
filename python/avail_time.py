import os.path
import pytz
import sys
import glob
import holidays
from datetime import datetime, timedelta, timezone, time
from collections import defaultdict

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Set display time zone (e.g., "America/New_York", "Europe/London", "Asia/Tokyo")
# List of valid time zones https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
DISPLAY_TIMEZONE = "Asia/Tokyo"
JP_HOLIDAYS = holidays.country_holidays("JP")
START_TIME = time(hour=9, minute=0)
END_TIME = time(hour=20, minute=30)
QUERY_RANGE = timedelta(days=30)
MIN_SLOT_DURATION = timedelta(hours=0, minutes=30)

# OAuth scope - read-only access to Calendars is sufficient for free/busy query
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

# Calendar IDs to check (replace with your three calendars' IDs or use 'primary')
calendar_ids = [
    "rioyokota@rio.scrc.iir.isct.ac.jp",
    "rio.gsic.titech.ac.jp_igf61u54oabk6gt9uqnv60avlk@group.calendar.google.com",
    "rio.gsic.titech.ac.jp_fou7rom2g1kd7tinua0dos6920@group.calendar.google.com"
]

def main():
    """Find and print free time slots by combining busy times from multiple calendars."""
    creds = None
    # Load saved user credentials from token.json if available
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If no valid credentials, authenticate via OAuth2 flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            credential_files = glob.glob("credentials*.json")
            if credential_files:
                flow = InstalledAppFlow.from_client_secrets_file(credential_files[0], SCOPES)
            else:
                raise FileNotFoundError("No credentials file matching 'credentials*.json' found.")
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token_file:
            token_file.write(creds.to_json())

    # Build the Google Calendar API service
    service = build("calendar", "v3", credentials=creds)

    # Define the time range: from now (today) to one month from now, in UTC
    now_utc = datetime.now(tz=timezone.utc)
    time_min = now_utc.isoformat()  # RFC3339 timestamp for start (inclusive)
    time_max = (now_utc + QUERY_RANGE).isoformat()  # end (exclusive)
    # Prepare the request body for freeBusy.query
    body = {
        "timeMin": time_min,
        "timeMax": time_max,
        "items": [{"id": cal_id} for cal_id in calendar_ids]
        # "timeZone": "UTC"  # optional: specify a time zone for the output times
    }
    # Query the free/busy info for the calendars
    response = service.freebusy().query(body=body).execute()
    # The response contains busy times for each queried calendar
    calendars_busy = response.get("calendars", {})  # dictionary with calendarId keys
    
    # Collect all busy intervals from all calendars
    busy_intervals = []
    for cal_id, cal_info in calendars_busy.items():
        for busy in cal_info.get("busy", []):
            # Parse the start and end times of each busy interval
            start_str = busy["start"]
            end_str   = busy["end"]
            # Convert to datetime objects (parse RFC3339 strings)
            # Ensure we handle 'Z' timezone (UTC) by replacing it with +00:00 if present
            if start_str.endswith("Z"):
                start_str = start_str[:-1] + "+00:00"
            if end_str.endswith("Z"):
                end_str = end_str[:-1] + "+00:00"
            start_dt = datetime.fromisoformat(start_str)
            end_dt   = datetime.fromisoformat(end_str)
            busy_intervals.append((start_dt, end_dt))
    
    # If no busy intervals at all, then the entire range is free
    if not busy_intervals:
        free_slots = [(datetime.fromisoformat(time_min), datetime.fromisoformat(time_max))]
    else:
        # Sort all busy intervals by start time
        busy_intervals.sort(key=lambda x: x[0])
        # Merge overlapping or contiguous busy intervals across all calendars
        merged_busy = [busy_intervals[0]]
        for current_start, current_end in busy_intervals[1:]:
            last_start, last_end = merged_busy[-1]
            if current_start <= last_end:
                # If the current busy interval overlaps or touches the last one, merge them
                if current_end > last_end:
                    merged_busy[-1] = (last_start, current_end)
            else:
                # No overlap, this is a separate busy interval
                merged_busy.append((current_start, current_end))
        # Compute free intervals between merged busy intervals
        free_slots = []
        # Free time before the first busy interval
        overall_start = datetime.fromisoformat(time_min)
        overall_end   = datetime.fromisoformat(time_max)
        if overall_start < merged_busy[0][0]:
            free_slots.append((overall_start, merged_busy[0][0]))
        # Free times between busy intervals
        for i in range(len(merged_busy) - 1):
            end_current = merged_busy[i][1]
            start_next  = merged_busy[i+1][0]
            if end_current < start_next:
                free_slots.append((end_current, start_next))
        # Free time after the last busy interval
        if merged_busy[-1][1] < overall_end:
            free_slots.append((merged_busy[-1][1], overall_end))
    # Convert results to desired timezone
    try:
        target_tz = pytz.timezone(DISPLAY_TIMEZONE)
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"Error: Unknown timezone '{DISPLAY_TIMEZONE}'. Please check your timezone.")
        sys.exit(1)

    print(f"Free time slots ({DISPLAY_TIMEZONE}) from {now_utc.date()} to {(now_utc + timedelta(days=30)).date()}:")

    daily_slots = defaultdict(list)

    for start, end in free_slots:
        start_local = start.astimezone(target_tz)
        end_local = end.astimezone(target_tz)
     
        current = start_local
        while current < end_local:
            if current.weekday() < 5 and current.date() not in JP_HOLIDAYS: # Monday=0, Sunday=6; limit to weekdays and avoid holidays
                slot_start = max(current.replace(hour=START_TIME.hour, minute=START_TIME.minute, second=0, microsecond=0), current)
                slot_end = current.replace(hour=END_TIME.hour, minute=END_TIME.minute, second=0, microsecond=0)

                if slot_start < slot_end and slot_start < end_local and (min(slot_end, end_local) - slot_start) >= MIN_SLOT_DURATION:
                    final_start = slot_start
                    final_end = min(slot_end, end_local)
                    if (final_end - final_start) >= MIN_SLOT_DURATION:
                        date_key = final_start.strftime('%m/%d (%a)')
                        slot_str = f"{final_start.strftime('%H:%M')}–{final_end.strftime('%H:%M')}"
                        daily_slots[date_key].append(slot_str)

            current = (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    for date, slots in sorted(daily_slots.items()):
        slots_str = ", ".join(slots)
        print(f"{date} {slots_str}")

if __name__ == "__main__":
    main()
