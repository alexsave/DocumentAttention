import collections
import json
import pickle
from collections import defaultdict
import os
import hashlib
import tempfile

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser

from common import ChatHistory, RetrievalHandler, TimerLogger, chunkenize, chunkenize_smalloverlap, llm, loadfiles, chunk_size_bytes, LLM_MODEL

EMBED_MODEL = 'nomic-embed-text'

preprocessing_timer = TimerLogger("Preprocessing")

corpus_size = 0

INFO_FILE = "info.pkl"
GEOCODE_CACHE_FILE = "geocode_cache.json"

info_store = {}
chunk_store = {}
geocode_cache = {}

# Load geocode cache if it exists
if os.path.exists(GEOCODE_CACHE_FILE):
    try:
        with open(GEOCODE_CACHE_FILE, 'r') as f:
            geocode_cache = json.load(f)
        print(f"Loaded {len(geocode_cache)} locations from geocode cache.")
    except json.JSONDecodeError:
        print("Error loading geocode cache. Starting with empty cache.")
        geocode_cache = {}
else:
    print("No geocode cache found. Starting with empty cache.")

loaded_files = loadfiles()

show_year = False

# Compute a hash to verify the state of the input files
hash_input = pickle.dumps([chunk_size_bytes, LLM_MODEL])
hash_value = hashlib.sha256(hash_input).hexdigest()

save_file = f"{hash_value[:7]}-{INFO_FILE}"

# Function to save progress using pickle
def save_progress():
    with tempfile.NamedTemporaryFile('wb', delete=False) as temp_file:
        pickle.dump({"hash": hash_value, "info_store": info_store}, temp_file)
        temp_file_path = temp_file.name
    os.replace(temp_file_path, save_file)

# Function to save geocode cache
def save_geocode_cache():
    with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
        json.dump(geocode_cache, temp_file)
        temp_file_path = temp_file.name
    os.replace(temp_file_path, GEOCODE_CACHE_FILE)
    print(f"Saved {len(geocode_cache)} locations to geocode cache.")

# Load sentiment data from file if it exists and matches the hash
if os.path.exists(save_file):
    with open(save_file, 'rb') as f:
        try:
            saved_data = pickle.load(f)
            if saved_data.get("hash") == hash_value:
                info_store = saved_data["info_store"]
                print("Loaded existing sentiment data from file.")
            else:
                print("Sentiment file found but hash mismatch. Starting fresh.")
        except pickle.PickleError:
            print("Error decoding sentiment file. Starting fresh.")
else:
    print("No existing sentiment file found. Starting fresh.")

# Location mapping dictionary to standardize location names
LOCATION_MAPPING = {
    # San Francisco neighborhoods
    "marina green": "San Francisco",
    "chase center": "San Francisco",
    "maritime social": "San Francisco",
    "mission in soma": "San Francisco",
    "harrison": "San Francisco",
    "monroe sf": "San Francisco",
    "oak st": "San Francisco",
    "bayview": "San Francisco",
    "tenderloin": "San Francisco",
    "dolores area": "San Francisco",
    "buena vista": "San Francisco",
    "bi rite": "San Francisco",
    "spear office": "San Francisco",
    "omn": "San Francisco",
    "tenderloin area": "San Francisco",
    "fillmore st": "San Francisco",
    "valencia st": "San Francisco",
    "presidio": "San Francisco",
    "pacific ave": "San Francisco",
    "spicy king": "San Francisco",
    "west wood": "San Francisco",
    "sf": "San Francisco",
    "hayes valley": "San Francisco",
    "marina": "San Francisco",
    "soma": "San Francisco",
    "embarcadero": "San Francisco",
    "pac heights": "San Francisco",
    "golden gate park": "San Francisco",
    "fillmore": "San Francisco",
    "gatekey": "San Francisco",
    "fogo de chao": "San Francisco",
    "sfo office": "San Francisco",
    "blakes": "San Francisco",
    "potrero hill": "San Francisco",
    "beale st": "San Francisco",
    "fishermans wharf": "San Francisco",
    "mission": "San Francisco",
    "laurel heights": "San Francisco",
    "spe121 office": "San Francisco",
    "japantown": "San Francisco",
    "16th st station": "San Francisco",
    "steiner and haight": "San Francisco",
    "civic center bart": "San Francisco",
    "dolores park": "San Francisco",
    "market and church": "San Francisco",
    "natoma cabana": "San Francisco",
    "california st": "San Francisco",
    "castro muni stop": "San Francisco",
    "trader joes on spring st": "San Francisco",
    "monroes": "San Francisco",
    "spe office": "San Francisco",
    "rincon hill": "San Francisco",
    "valencia and 14th": "San Francisco",
    "potrero goodwill": "San Francisco",
    "balboa park": "San Francisco",
    "16th between valencia and guerrero": "San Francisco",
    "baker beach": "San Francisco",
    "haight": "San Francisco",
    "mission dolores area": "San Francisco",
    "tc5 gym": "Sunnyvale",
    "salesforce transit center": "San Francisco",
    "sutro open space": "San Francisco",
    "bay view": "San Francisco",
    "tc5": "San Francisco", 
    "noe valley": "San Francisco",
    "the city by the bay": "San Francisco",
    "san francisco bayview": "San Francisco",
    "bill graham auditorium": "San Francisco",
    "bi-rite": "San Francisco",
    "mission neighborhood": "San Francisco",
    "1015 folsom": "San Francisco",
    "kezar stadium": "San Francisco",
    "the mission": "San Francisco",
    "richmond": "San Francisco",
    "ocean beach": "San Francisco",
    "soma square": "San Francisco",
    "ikea": "San Francisco",

    "east coast": "none",
    "city": "none", # too vague
    "downtown": "none", # too vague
    "main st": "none", # too vague
    "main st.": "none", # too vague
    "7 miles from the author's place": "none", # too vague
    "nearby": "none", # too vague
    "the city": "none", # too vague
    "north": "none", # too vague

    "pac pipes": "Oakland",


    "heavenly": "Lake Tahoe",


    "concord": "Concord, NH",
    "broadway cafe": "Concord, NH",
    "st. paul's": "Concord, NH",
    "white park": "Concord, NH",
    "beaver meadow school": "Concord, NH",
    "129 fisherville rd": "Concord, NH",
    "whites": "Concord, NH",
    "the fisk loop": "Concord, NH",
    "ferrin rd": "Concord, NH",
    "concord nh": "Concord, NH",
    
    "mill": "Durham, NH",
    "durham": "Durham, NH",
    "mill house": "Durham, NH",
    "mill street": "Durham, NH",
    "unh": "Durham, NH",
    "madcom": "Durham, NH",
    "chloes lodge apartment": "Durham, NH",
    "durham nh": "Durham, NH",
    "hoco": "Durham, NH",
    "dover, nh": "Dover, NH",
    "dover nh": "Dover, NH",

    "hudson yards": "NYC",
    "williamsburg": "NYC",
    "harlem": "NYC",
    "east village": "NYC",
    "west village": "NYC",
    "new york city": "NYC",
    "manhattan": "NYC",
    "long island": "NYC",
    "flushing": "NYC",
    "hudson square": "NYC",
    "high line": "NYC",
    "3rd avenue": "NYC",
    "bank st #1g": "NYC",
    "downtown brooklyn": "NYC",
    "dumbo": "NYC",
    "penn station": "NYC",
    "upper west side": "NYC",
    "brooklyn heights": "NYC",
    "bryant park": "NYC",
    "kips bay": "NYC",
    "dumbo": "NYC",
    "flatiron equinox": "NYC",
    "pearl studios": "NYC",
    "times square": "NYC",
    "ues": "NYC",
    "empire state building and world trade center": "NYC",
    "ues ralph lauren store": "NYC",
    "riverside dr": "NYC",
    "8th floor gym": "NYC",
    "macy's": "NYC",
    "soho": "NYC",
    "battery park city": "NYC",
    "world trade center": "NYC",
    "bedford ave station": "NYC",
    "bronx": "NYC",
    "red hook": "NYC",
    "schott": "NYC",

    "cambridge": "Cambridge, MA",
    "epsom": "Epsom, NH",

    "nh": "New Hampshire",
    "long pond road": "Concord, NH",
    "east concord": "Concord, NH",
    "white's park": "Concord, NH",
    "memorial field": "Concord, NH",
    "colleens house": "Concord, NH",
    "chs and commons b": "Concord, NH",
    "nichols": "Concord, NH",
    "st. paul": "Concord, NH",
    "cap city": "Concord, NH",

    "gould hill": "Hopkinton, NH",


    "decodance": "San Francisco",

    "hmblt3": "Sunnyvale",

    "SJT": "NYC",


    "manchester west": "Manchester, NH",
    "oyster river": "Durham, NH",
    "philbricks": "Durham, NH",
    "uedge": "Durham, NH",
    "ski house": "Durham, NH",
    "faculty neighborhood": "Durham, NH",
    "stoke 7th floor": "Durham, NH",
    "kingsbury": "Durham, NH",
    "gibbs hall": "Durham, NH",
    "the knot": "Durham, NH",
    "adams tower": "Durham, NH",
    "mast rd apartments": "Durham, NH",
    "dennison st": "Durham, NH",
    "horty hall": "Durham, NH",
    "red door": "Durham, NH",
    "47 main st, apt 10": "Durham, NH",
    "mast": "Durham, NH",
    "b lot": "Durham, NH",
    "libbys": "Durham, NH",
    "mast rd": "Durham, NH",

    "french laundry": "Yountville, CA",

    "bethlehem": "Bethlehem, NH",


    "capitol hill": "Seattle",

    "zedd concert": "Las Vegas",

    "broadway b": "Dover, NH",

    "dover": "Dover, NH",


    "the goat": "Portsmouth, NH",
    "portsmouth office": "Portsmouth, NH",


    "buffalo": "Buffalo, NY",

    "logan airport": "Boston",

    "popovers on the square": "Portsmouth, NH",

    "mines falls": "Nashua, NH",

    "bay rd": "Durham, NH",
    "scott": "Durham, NH",


    "downtown sunnyvale": "Sunnyvale",
    "california avenue": "Palo Alto",
    "monzus": "Santa Clara",

    "newbury st": "Boston",
    "south station": "Boston",
    "reggie lewis track center": "Boston",
    "boston university": "Boston",
    "boston logan": "Boston",

    "gangnam": "Seoul",
    "itaewon": "Seoul",
    "bugaksan": "Seoul",
    "hongdae": "Seoul",
    "seongsu": "Seoul",
    "myongdong": "Seoul",
    "myeongdong": "Seoul",

    "shinsaibashi": "Osaka",
    "osaka": "Osaka",

    "roppongi": "Tokyo",
    "shibuya crossing": "Tokyo",
    "ginza": "Tokyo",
    "ueno park": "Tokyo",
    "yodobashi akiba": "Tokyo",

    "osulloc tea museum": "Jeju",
    "busan": "Busan",
    "fuakata": "Japan",
    "okoyama": "Japan",

    "vegas": "Las Vegas",
    "venetian": "Las Vegas",
    "las vegas": "Las Vegas",

    "istanbul": "Istanbul",
    "turkey": "Turkey",
    "grand bazaar": "Istanbul",

    "moab": "Moab, UT",
    "salt lake city": "Salt Lake City",
    "vail": "Vail, CO",
    "leadville": "Leadville, CO",
    "yellowstone": "Yellowstone National Park",
    "glacier national park": "Glacier National Park",
    "olympic national park": "Olympic National Park",
    "mt. rainier national park": "Mt. Rainier National Park",
    "grand prismatic geyser": "Yellowstone National Park",
    "cascade canyon": "Grand Teton National Park",
    "seattle": "Seattle",
    "eugene, oregon": "Eugene, OR",
    "beverly hills": "Beverly Hills",
    "hollywood blvd": "Los Angeles",
    "glass beach": "Fort Bragg, CA",
    "aspen": "Aspen, CO",
    "buffalo": "Buffalo, NY",
    "atlantic city": "Atlantic City, NJ",
    "miami beach": "Miami Beach",
    "madrid": "Madrid",
    "seville": "Seville",
    "vatican basilica": "Rome",
    "westwood": "San Francisco",
    "rodeo drive": "Beverly Hills",
    "phoenix": "Phoenix, AZ",
    "west texas": "Texas",
    "lansing": "San Francisco",
    "virginia": "Virginia",
    "nevada": "Nevada",
    "lowell": "Lowell, MA",
    "colorado": "Colorado",
    "tampa": "Tampa, FL",
    "lax": "Los Angeles",

    "uvm": "Burlington, VT",
    "dartmouth": "Hanover, NH",
    "providence college": "Providence",
    "duke": "Durham, NC",
    "central connecticut state university": "New Britain, CT",
    "ucla": "Los Angeles",
    "umass lowell": "Lowell, MA",

    "la": "Los Angeles",
    "loon": "Lincoln, NH",
    "epping": "Epping, NH",
    "santa cruz": "Santa Cruz, CA",
    "ma": "Massachusetts",
    "manchester": "Manchester, NH",
    "yarmouth": "Yarmouth, MA",
    "nh": "New Hampshire",

    "shakespeare globe theatre": "London",

    "manch": "Manchester, NH",
    "philly": "Durham, NH",
    "central park": "NYC",
    "tribeca": "NYC",
    "greenwich ave": "NYC",
    "sjt": "NYC",
    "the veranda": "Durham, NH",
    "hampton beach": "Hampton Beach",
    " durham": "Durham, NH",
    "sunnyvale": "Sunnyvale",
    "sunnyvale, california": "Sunnyvale",

    "mast rd extension": "Durham, NH",

    "moc": "Londonderry, NH",
    "mocs": "Londonderry, NH",

    "bbq noodles": "San Francisco",
    "new britain": "New Britain, CT",
    "fidi": "NYC",
    "d3": "Concord, NH",
    "nike store": "Kittery, ME",
    "newmarket": "Newmarket, NH",
    "st. albans": "St. Albans, VT",
    "portsmouth": "Portsmouth, NH",
    "ramada": "Cape Cod",
    "pembroke": "Pembroke, NH",
    "cannon": "Franconia, NH",
    "alton": "Alton, NH",
    "mt. washington": "Sargent's Purchase, NH",
    "newfields": "Newfields, NH",
    "rye beach": "Rye, NH",
    "green room": "San Francisco",
    "page": "Page, AZ",
    "hampton": "Hampton, NH",
    "the bay area": "none",
    "new england": "none",
    "hopkinton": "Hopkinton, NH",
    "boulder field": "Durham, NH",
    "grant village": "Grant Village, WY",
    "forest": "Durham, NH",
    "salem": "Salem, NH",
    "financial district": "NYC",
    "south bay": "none",
    "whole foods": "none",


    "downtown area": "none",
}

def standardize_location(location):
    """Standardize location names using the mapping dictionary."""
    if not location or location.lower() == 'none':
        return 'none'
    
    location_lower = location.lower()
    location = LOCATION_MAPPING.get(location_lower, location)

    location_lower = location.lower()
    return LOCATION_MAPPING.get(location_lower, location)
    


# Function to extract sentiment score
def extract_location(chunk):

    prompt = f"""You are analysing a journal fragment.
Extract the location ONLY if the fragment clearly indicates being in some city/metropolitan area, do not include references to places.
If you're not sure, return a JSON with location none.
If you are not familar with the location, return a JSON with location none.
If the fragment is set in a street or a building or a neighborhood or something smaller than a city, return a JSON with the city of the fragment.
If the location is not specific, such as "home" or "work", return a JSON with location none.
Return ONE JSON per line.

Text:
{chunk}

Example Outputs:
{{"location": "Cape Cod"}}
{{"location": "none"}}
"""


    response, stats = llm(prompt, log=True, format="json")

    try:
        loc = json.loads(response.strip())
    except json.JSONDecodeError:
        loc = {"location":"none"}

    if 'location' not in loc:
        loc['location'] = "none"

    print(loc)
    return loc


# Function to parse date strings
def parse_date(date_str):
    try:
        # Use dateutil.parser to parse the date string
        return parser.parse(date_str)
    except (ValueError, parser.ParserError):
        return None

# Process chunks and extract sentiment scores
chunks_processed = 0
for info in loaded_files:
    date_str = info["date"]
    print(f"Original date string: {date_str}")
    parsed_date = parse_date(date_str)
    if parsed_date is None:
        print(f"Could not parse date: {date_str}")
        continue  # Skip this entry
    content = info["content"]
    corpus_size += len(content)

    chunks = chunkenize_smalloverlap(content, 8192)

    for i, chunk in enumerate(chunks):
        id = f"{date_str}#{i}"
        print(f"Processing chunk ID: {id}")

        chunk_store[id] = date_str + "\n" + chunk

        # Skip if already processed
        if id in info_store:
            if chunks_processed % 5 == 0:
                chunks_processed += 1
            continue

        chunks_processed += 1
        location = extract_location(chunk)

        info_store[id] = {
            'date_str': date_str,  # Store date string
            'date': parsed_date,   # Store parsed date
            #'chunk': chunk,
            'location': location["location"]
        }

        # Save progress every 50 chunks
        if chunks_processed % 50 == 0:
            save_progress()

# Save the final progress
save_progress()

preprocessing_timer.stop_and_log(corpus_size)

spans = defaultdict(list)


li = []


sorted_keys = sorted(info_store.keys())

# a bit tricky because it's weekly rather than daily
# best bet might be to just determine one location for each week
# all per-chunk locations should have the same v["date"], so we could build {date -> location}
for k in sorted_keys:
    v = info_store[k]

    location = standardize_location(v["location"])

    if not li:
        li.append({"location": location, "start": v["date"], "end": v["date"] + timedelta(days=7)})

    else:
        last = li[-1]
        cur_date = v["date"]
        if location == "none" or last["location"] == location:
            if cur_date > last["end"]:
                last["end"] = cur_date
        else:
            li.append({"location": location, "start": cur_date, "end": cur_date + timedelta(days=7)})
    print(v)



records = []
for location,lst in spans.items():
    for start,end,_ in lst:
        if end is None:
            end = datetime.now().isoformat()
        records.append({"location": location, "start": start, "end": end})

df = pd.DataFrame(li)
print(df)

# Calculate total duration for each location
location_durations = {}
for _, row in df.iterrows():
    location = row["location"]
    duration = (row["end"] - row["start"]).days
    if location in location_durations:
        location_durations[location] += duration
    else:
        location_durations[location] = duration

# Sort locations by total duration (descending)
sorted_locations = sorted(location_durations.items(), key=lambda x: x[1], reverse=True)
row_map = {location: i for i, (location, _) in enumerate(sorted_locations)}
df["row"] = df["location"].map(row_map)

df["start"] = pd.to_datetime(df["start"])
df["end"] = pd.to_datetime(df["end"])

# Plot
fig, ax = plt.subplots(figsize=(12, min(20, 1 + len(row_map) * 0.3)))  # Limit height to 20 inches
colors = {c: plt.cm.tab20(i % 20) for i,c in enumerate(row_map)}
for _, r in df.iterrows():
    ax.barh(y=r["row"],
            left=r["start"],
            width = (r["end"] - r["start"]).days,


            height=0.6,
            color=colors[r["location"]],
            label=r["location"])

ax.set_yticks(list(row_map.values()))
ax.set_yticklabels(list(row_map.keys()))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

# Single legend entry per location
handles, labels = ax.get_legend_handles_labels()
uniq = dict(zip(labels, handles))
ax.legend(uniq.values(), uniq.keys(), fontsize="small", ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.15))
ax.set_title("Location Timeline (from info_store)")

# Format x-axis as dates
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.autofmt_xdate()

# 7. Label and show
ax.set_xlabel("Date")

# Adjust layout to prevent cutoff
plt.tight_layout()

# Print locations that have only 1 span
location_span_counts = df['location'].value_counts()
single_span_locations = location_span_counts[location_span_counts == 1].index.tolist()
print("\nLocations with only 1 span:")
for location in single_span_locations:
    span_row = df[df['location'] == location].iloc[0]
    duration = (span_row['end'] - span_row['start']).days
    print(f"{location}: {span_row['start'].strftime('%Y-%m-%d')} to {span_row['end'].strftime('%Y-%m-%d')} ({duration} days)")

#plt.show()

def create_location_map(df):
    # Create a map centered at a default location
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    # Initialize geocoder
    geolocator = Nominatim(user_agent="my_agent")
    
    # Get unique locations and their coordinates
    print("Getting unique locations and their coordinates")
    locations = {}
    for location in df['location'].unique():
        if location.lower() != 'none':
            if location in geocode_cache:
                # Use cached coordinates
                locations[location] = geocode_cache[location]
                print(f"Using cached coordinates for {location}: {locations[location]}")
            else:
                try:
                    # Add a small delay to respect geocoding service limits
                    time.sleep(1)
                    location_data = geolocator.geocode(location)
                    if location_data:
                        coords = (location_data.latitude, location_data.longitude)
                        locations[location] = coords
                        # Cache the coordinates
                        geocode_cache[location] = coords
                        print(f"Location: {location}, Coordinates: {coords}")
                except GeocoderTimedOut:
                    print(f"Timeout for location: {location}")
                    continue
    
    # Save the geocode cache
    save_geocode_cache()
    
    # Sort DataFrame by start date
    df_sorted = df.sort_values('start')
    
    # Add markers and lines for each location
    prev_coords = None
    for _, row in df_sorted.iterrows():
        if row['location'].lower() != 'none' and row['location'] in locations:
            coords = locations[row['location']]
            
            # Add marker
            folium.Marker(
                coords,
                popup=f"{row['location']}<br>{row['start'].strftime('%Y-%m-%d')}",
                tooltip=row['location']
            ).add_to(m)
            
            # Add line if there's a previous location
            if prev_coords:
                folium.PolyLine(
                    [prev_coords, coords],
                    color='blue',
                    weight=2,
                    opacity=0.8
                ).add_to(m)
            
            prev_coords = coords
    
    # Save the map
    m.save('location_map.html')
    print("Map has been saved as 'location_map.html'")

# After creating the DataFrame, add this line to create the map
create_location_map(df)

