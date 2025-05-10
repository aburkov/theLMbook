import os
import json
import random
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SCOPES = "user-library-read playlist-modify-public playlist-read-private playlist-read-collaborative"
NEW_PLAYLIST_NAME = "New Gemini Recommendations"
ALL_RECS_PLAYLIST_NAME = "All Gemini Recommendations"

TARGET_NEW_SONGS_COUNT = 20
MAX_GEMINI_ATTEMPTS = 10 # Increased attempts as we ask for exactly 20 each time
MAX_SONGS_TO_GEMINI_PROMPT = 200 # Max liked songs for the initial Gemini prompt

GEMINI_MODEL = "google/gemini-2.5-flash-preview"
# Ensure this model ID is valid. If not, try "google/gemini-flash-1.5"

# --- Helper Functions ---

def get_spotify_client():
    auth_manager = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPES,
        cache_path=".spotify_cache"
    )
    sp = spotipy.Spotify(auth_manager=auth_manager)
    print("Successfully authenticated with Spotify.")
    return sp

def get_all_liked_songs_details(sp):
    print("Fetching all liked songs details...")
    liked_songs_details = []
    offset = 0
    limit = 50
    while True:
        try:
            results = sp.current_user_saved_tracks(limit=limit, offset=offset)
            if not results or not results['items']:
                break
            for item in results['items']:
                track = item.get('track')
                if track and track.get('name') and track.get('artists'):
                    if track['artists']: # Ensure artist list is not empty
                        artist_name = track['artists'][0]['name']
                        liked_songs_details.append({"track": track['name'], "artist": artist_name})
            offset += limit
            print(f"Fetched {len(liked_songs_details)} liked songs so far...")
            if not results.get('next'):
                break
            time.sleep(0.05)
        except Exception as e:
            print(f"Error fetching liked songs page: {e}")
            break
    print(f"Total liked songs details fetched: {len(liked_songs_details)}")
    return liked_songs_details

def get_playlist_by_name(sp, playlist_name, user_id):
    playlists = sp.current_user_playlists(limit=50)
    while playlists:
        for playlist in playlists['items']:
            if playlist['name'] == playlist_name and playlist['owner']['id'] == user_id:
                return playlist
        if playlists['next']:
            playlists = sp.next(playlists)
            time.sleep(0.05)
        else:
            playlists = None
    return None

def get_or_create_playlist_id(sp, user_id, playlist_name, public=True):
    playlist_object = get_playlist_by_name(sp, playlist_name, user_id)
    if playlist_object:
        print(f"Found existing playlist: '{playlist_name}' (ID: {playlist_object['id']})")
        return playlist_object['id']
    else:
        print(f"Playlist '{playlist_name}' not found. Creating it...")
        try:
            new_playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=public)
            print(f"Successfully created playlist: '{playlist_name}' (ID: {new_playlist['id']})")
            return new_playlist['id']
        except Exception as e:
            print(f"Error creating playlist '{playlist_name}': {e}")
            return None

def get_playlist_tracks_simplified(sp, playlist_id):
    if not playlist_id: return []
    print(f"Fetching tracks from playlist ID: {playlist_id}...")
    playlist_tracks = []
    offset = 0
    limit = 100
    while True:
        try:
            results = sp.playlist_items(playlist_id, limit=limit, offset=offset, fields="items(track(name,artists(name))),next")
            if not results or not results['items']: break
            for item in results['items']:
                track_info = item.get('track')
                if track_info and track_info.get('name') and track_info.get('artists'):
                    if track_info['artists']:
                        artist_name = track_info['artists'][0]['name']
                        playlist_tracks.append({"track": track_info['name'], "artist": artist_name})
            offset += limit
            print(f"Fetched {len(playlist_tracks)} tracks from playlist ID {playlist_id} so far...")
            if not results.get('next'): break
            time.sleep(0.05)
        except Exception as e:
            print(f"Error fetching playlist items for {playlist_id}: {e}")
            break
    print(f"Total tracks fetched from playlist ID {playlist_id}: {len(playlist_tracks)}")
    return playlist_tracks

def get_gemini_recommendations(api_key, conversation_history):
    """
    Sends the conversation history to Gemini and requests recommendations.
    Returns a tuple: (parsed_recommendations_list, raw_assistant_response_content_string)
    The last message in conversation_history is assumed to be the current user prompt.
    """
    print(f"\nSending request to Gemini with {len(conversation_history)} messages in history...")
    if not conversation_history or conversation_history[-1]["role"] != "user":
        print("Error: Conversation history is empty or does not end with a user message.")
        return [], None

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": GEMINI_MODEL,
                "messages": conversation_history,
                "response_format": {"type": "json_object"}
            },
            timeout=60 # Increased timeout for potentially longer LLM responses
        )
        response.raise_for_status()
        
        response_data = response.json()
        raw_assistant_response_content = response_data['choices'][0]['message']['content']
        
        recommendations = []
        try:
            parsed_content = json.loads(raw_assistant_response_content)
            if isinstance(parsed_content, list):
                recommendations = parsed_content
            elif isinstance(parsed_content, dict) and len(parsed_content.keys()) == 1:
                key = list(parsed_content.keys())[0]
                if isinstance(parsed_content[key], list):
                    recommendations = parsed_content[key]
        except json.JSONDecodeError:
            print("Gemini response was not directly parsable JSON. Attempting to clean...")
            content_to_parse = raw_assistant_response_content
            if content_to_parse.startswith("```json"): content_to_parse = content_to_parse[7:]
            if content_to_parse.endswith("```"): content_to_parse = content_to_parse[:-3]
            content_to_parse = content_to_parse.strip()
            try:
                parsed_content = json.loads(content_to_parse)
                if isinstance(parsed_content, list): recommendations = parsed_content
                elif isinstance(parsed_content, dict) and len(parsed_content.keys()) == 1:
                    key = list(parsed_content.keys())[0]
                    if isinstance(parsed_content[key], list): recommendations = parsed_content[key]
            except json.JSONDecodeError as e_clean:
                print(f"Error: Gemini response could not be parsed as JSON even after cleaning: {e_clean}")
                print(f"Gemini Raw Response Content:\n{raw_assistant_response_content}")
                return [], raw_assistant_response_content # Return raw content for history even on parse error

        valid_recommendations = []
        for rec in recommendations:
            if isinstance(rec, dict) and "track" in rec and "artist" in rec:
                valid_recommendations.append({"track": str(rec["track"]), "artist": str(rec["artist"])})
            else:
                print(f"Warning: Skipping invalid recommendation format from Gemini: {rec}")
        
        print(f"Received {len(valid_recommendations)} validly structured recommendations from Gemini.")
        return valid_recommendations, raw_assistant_response_content

    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenRouter API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            try: print(f"Response content: {e.response.json()}")
            except json.JSONDecodeError: print(f"Response content: {e.response.text}")
        return [], None
    except (KeyError, IndexError) as e:
        raw_resp_text = response.text if 'response' in locals() else 'No response object'
        print(f"Error parsing Gemini response structure: {e}")
        print(f"Gemini Raw Response (full): {raw_resp_text}")
        return [], None


def verify_songs_on_spotify_v2(sp, recommended_songs_details):
    print("\nVerifying recommended songs on Spotify...")
    available_songs_info = []
    for song_detail in recommended_songs_details:
        track_name = song_detail.get('track')
        artist_name = song_detail.get('artist')
        if not track_name or not artist_name: continue
        query = f"track:{track_name} artist:{artist_name}"
        try:
            results = sp.search(q=query, type="track", limit=1)
            time.sleep(0.05)
            if results and results['tracks']['items']:
                found_track = results['tracks']['items'][0]
                available_songs_info.append({
                    "uri": found_track['uri'],
                    "track": found_track['name'],
                    "artist": found_track['artists'][0]['name']
                })
                print(f"  Found on Spotify: '{found_track['name']}' by {found_track['artists'][0]['name']}")
            else:
                print(f"  Not found on Spotify: '{track_name}' by {artist_name}")
        except Exception as e:
            print(f"  Error searching for '{track_name}' by {artist_name}: {e}")
    print(f"\nVerified {len(available_songs_info)} songs as available on Spotify.")
    return available_songs_info

def update_playlist_items(sp, playlist_id, track_uris, replace=False):
    if not playlist_id: return False
    if not track_uris and not replace: return True
    if not track_uris and replace:
        try:
            sp.playlist_replace_items(playlist_id, [])
            print(f"Cleared all items from playlist ID {playlist_id}.")
            return True
        except Exception as e: print(f"Error clearing playlist {playlist_id}: {e}"); return False

    action = "Replacing" if replace else "Adding"
    print(f"{action} {len(track_uris)} songs for playlist ID {playlist_id}...")
    try:
        if replace:
            # Spotipy's playlist_replace_items handles batching internally up to 100.
            # For >100, it might still be one call to Spotify API that errors,
            # or spotipy might make multiple calls.
            # Let's stick to safer manual batching if >100 for replace.
            if len(track_uris) <= 100:
                 sp.playlist_replace_items(playlist_id, track_uris)
            else:
                sp.playlist_replace_items(playlist_id, []) # Clear
                for i in range(0, len(track_uris), 100):
                    sp.playlist_add_items(playlist_id, track_uris[i:i + 100])
                    time.sleep(0.1)
        else: # Appending
            for i in range(0, len(track_uris), 100):
                sp.playlist_add_items(playlist_id, track_uris[i:i + 100])
                time.sleep(0.1)
        print(f"Successfully {action.lower()}ed songs in playlist ID {playlist_id}.")
        return True
    except Exception as e:
        print(f"Error {action.lower()}ing songs in playlist {playlist_id}: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    if not all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI, OPENROUTER_API_KEY]):
        print("Error: Missing environment variables. Please check .env file."); exit(1)

    sp_client = get_spotify_client()
    if not sp_client: exit(1)

    user_info = sp_client.me()
    user_id = user_info['id']
    print(f"Logged in as: {user_info.get('display_name', user_id)}")

    # 1. Get ALL liked songs and create a set for filtering
    all_my_liked_songs_details = get_all_liked_songs_details(sp_client)
    if not all_my_liked_songs_details:
        print("No liked songs found. Exiting."); exit()
    
    all_my_liked_songs_set = set()
    for song_detail in all_my_liked_songs_details:
        track = song_detail.get('track', "").strip().lower()
        artist = song_detail.get('artist', "").strip().lower()
        if track and artist:
            all_my_liked_songs_set.add((track, artist))
    print(f"Created set of {len(all_my_liked_songs_set)} unique liked songs for de-duplication.")

    # 2. Shuffle liked songs and take a sample for the initial Gemini prompt
    random.shuffle(all_my_liked_songs_details)
    sample_liked_songs_for_gemini_prompt = all_my_liked_songs_details[:MAX_SONGS_TO_GEMINI_PROMPT]

    # Get "All Gemini Recommendations" playlist history
    all_recs_playlist_id = get_or_create_playlist_id(sp_client, user_id, ALL_RECS_PLAYLIST_NAME)
    existing_all_recs_songs_details = []
    if all_recs_playlist_id:
        existing_all_recs_songs_details = get_playlist_tracks_simplified(sp_client, all_recs_playlist_id)
    
    all_recs_history_set = set() # Stores (track.lower(), artist.lower()) from "All Gemini Recs" playlist
    for song_detail in existing_all_recs_songs_details:
        track = song_detail.get('track', "").strip().lower()
        artist = song_detail.get('artist', "").strip().lower()
        if track and artist:
            all_recs_history_set.add((track, artist))
    print(f"Found {len(all_recs_history_set)} unique songs in '{ALL_RECS_PLAYLIST_NAME}' history.")

    # 3-5. Iteratively get new recommendations
    collected_new_songs_for_playlist_uris = []
    collected_new_songs_for_playlist_details = [] # Stores Spotify-verified dicts for final playlist

    conversation_history = []
    # Stores dicts {track, artist} of ALL songs Gemini suggests in this session (raw names from Gemini)
    # Used to tell Gemini what to avoid in follow-up prompts.
    all_gemini_suggestions_this_session_raw_details = [] 

    # Initial user prompt for the very first message to Gemini
    liked_songs_prompt_str = "\n".join([f"- \"{s['track']}\" by {s['artist']}" for s in sample_liked_songs_for_gemini_prompt])
    initial_user_prompt_content = f"""You are a music recommendation assistant. I will provide you with a list of songs I like.
Based on this list, please recommend {TARGET_NEW_SONGS_COUNT} additional songs that I might enjoy.
It's important that your response is ONLY a valid JSON array of objects, where each object has a "track" key (song title) and an "artist" key (artist name).
Do not include any other text, explanations, or markdown formatting outside of the JSON array.

Here are some songs I like:
{liked_songs_prompt_str}

Please provide {TARGET_NEW_SONGS_COUNT} new song recommendations in the specified JSON format."""
    conversation_history.append({"role": "user", "content": initial_user_prompt_content})


    for attempt in range(MAX_GEMINI_ATTEMPTS):
        if len(collected_new_songs_for_playlist_uris) >= TARGET_NEW_SONGS_COUNT:
            print("\nTarget number of new songs reached.")
            break

        print(f"\n--- Gemini Request Attempt {attempt + 1}/{MAX_GEMINI_ATTEMPTS} ---")
        
        # If this is not the first attempt, construct and add follow-up user message
        if attempt > 0:
            songs_suggested_by_gemini_this_session_str = "\n".join(
                [f"- \"{s['track']}\" by {s['artist']}" for s in all_gemini_suggestions_this_session_raw_details]
            )
            if not songs_suggested_by_gemini_this_session_str:
                songs_suggested_by_gemini_this_session_str = "(None previously suggested in this session)"

            follow_up_user_prompt_content = f"""Okay, thank you. Now, please provide {TARGET_NEW_SONGS_COUNT} MORE unique song recommendations based on the initial list of songs I like (provided at the start of our conversation).
It is very important that these new recommendations are different from any songs you've already suggested to me in this conversation. For reference, here are the songs you've suggested so far (please avoid these):
{songs_suggested_by_gemini_this_session_str}

Also, ensure these new recommendations are different from the initial list of liked songs I provided.
Your response must be ONLY a valid JSON array of objects, with "track" and "artist" keys, as before."""
            conversation_history.append({"role": "user", "content": follow_up_user_prompt_content})
            # Prune conversation history if it gets too long (optional, depends on model limits)
            # For now, let it grow for a few turns. Gemini Flash has a decent context window.
            # if len(conversation_history) > 10: # Example: keep last 10 messages + initial prompt
            #     conversation_history = [conversation_history[0]] + conversation_history[-9:]


        gemini_batch_recs_parsed, raw_assistant_response_str = get_gemini_recommendations(
            OPENROUTER_API_KEY,
            conversation_history
        )

        if raw_assistant_response_str: # If Gemini responded, add its response to history
            conversation_history.append({"role": "assistant", "content": raw_assistant_response_str})
        
        if not gemini_batch_recs_parsed:
            print("Gemini returned no valid recommendations in this batch or there was an API error.")
            if attempt < MAX_GEMINI_ATTEMPTS - 1: time.sleep(3)
            continue

        # Add raw suggestions from this Gemini batch to `all_gemini_suggestions_this_session_raw_details`
        # This list helps construct the "avoid these" part of the next follow-up prompt.
        for rec in gemini_batch_recs_parsed: # rec is dict {track, artist}
            all_gemini_suggestions_this_session_raw_details.append(rec)
        
        print(f"Gemini suggested {len(gemini_batch_recs_parsed)} songs. Verifying on Spotify and filtering...")
        verified_spotify_songs_this_batch = verify_songs_on_spotify_v2(sp_client, gemini_batch_recs_parsed)
        
        newly_added_this_turn_count = 0
        for verified_song_info in verified_spotify_songs_this_batch: # dict {'uri', 'track', 'artist'}
            if len(collected_new_songs_for_playlist_uris) >= TARGET_NEW_SONGS_COUNT:
                break

            # Use Spotify's canonical track/artist names for consistent checking
            spotify_track_name_lower = verified_song_info['track'].lower()
            spotify_artist_name_lower = verified_song_info['artist'].lower()
            spotify_song_key = (spotify_track_name_lower, spotify_artist_name_lower)
            
            is_liked = spotify_song_key in all_my_liked_songs_set
            is_in_all_recs_playlist_history = spotify_song_key in all_recs_history_set
            
            # Check if URI is already in the list we are building this session
            is_already_collected_for_new_playlist_this_session = any(
                vs['uri'] == verified_song_info['uri'] for vs in collected_new_songs_for_playlist_details
            )

            if not is_liked and not is_in_all_recs_playlist_history and not is_already_collected_for_new_playlist_this_session:
                collected_new_songs_for_playlist_uris.append(verified_song_info['uri'])
                collected_new_songs_for_playlist_details.append(verified_song_info)
                newly_added_this_turn_count +=1
                print(f"  ++ Collected for new playlist: '{verified_song_info['track']}' by '{verified_song_info['artist']}'")
            else:
                reason = []
                if is_liked: reason.append("is liked")
                if is_in_all_recs_playlist_history: reason.append("in all_recs history")
                if is_already_collected_for_new_playlist_this_session: reason.append("already collected this session")
                print(f"  -- Skipped '{verified_song_info['track']}' by '{verified_song_info['artist']}' (Reason: {', '.join(reason)})")

        print(f"Added {newly_added_this_turn_count} new songs this turn.")
        print(f"Total collected for new playlist so far: {len(collected_new_songs_for_playlist_uris)}/{TARGET_NEW_SONGS_COUNT}")
        
        if len(collected_new_songs_for_playlist_uris) >= TARGET_NEW_SONGS_COUNT:
            break 
        elif attempt < MAX_GEMINI_ATTEMPTS -1 :
            time.sleep(2) # Pause before next Gemini attempt

    # --- End of iterative collection ---

    final_uris_for_new_playlist = collected_new_songs_for_playlist_uris[:TARGET_NEW_SONGS_COUNT]
    final_details_for_all_recs_update = collected_new_songs_for_playlist_details[:TARGET_NEW_SONGS_COUNT]

    if not final_uris_for_new_playlist:
        print("\nNo new, verifiable songs were collected from Gemini after all attempts. Exiting.")
        exit()

    print(f"\nCollected {len(final_uris_for_new_playlist)} final new songs for '{NEW_PLAYLIST_NAME}'.")

    # 5. Save to "New Gemini Recommendations" (replacing)
    new_playlist_id = get_or_create_playlist_id(sp_client, user_id, NEW_PLAYLIST_NAME)
    if new_playlist_id:
        print(f"\nUpdating playlist '{NEW_PLAYLIST_NAME}' by replacing items...")
        if update_playlist_items(sp_client, new_playlist_id, final_uris_for_new_playlist, replace=True):
            playlist_url_new = sp_client.playlist(new_playlist_id)['external_urls']['spotify']
            print(f"Successfully updated '{NEW_PLAYLIST_NAME}'. URL: {playlist_url_new}")
    else:
        print(f"Could not create or find playlist '{NEW_PLAYLIST_NAME}'.")

    # 6. Add these songs to "All Gemini Recommendations" (appending)
    if all_recs_playlist_id and final_details_for_all_recs_update: # Use details to get URIs
        uris_to_add_to_all_recs = [song['uri'] for song in final_details_for_all_recs_update]
        print(f"\nAppending {len(uris_to_add_to_all_recs)} songs to '{ALL_RECS_PLAYLIST_NAME}'...")
        if update_playlist_items(sp_client, all_recs_playlist_id, uris_to_add_to_all_recs, replace=False):
            playlist_url_all = sp_client.playlist(all_recs_playlist_id)['external_urls']['spotify']
            print(f"Successfully appended songs to '{ALL_RECS_PLAYLIST_NAME}'. URL: {playlist_url_all}")
    elif not all_recs_playlist_id:
         print(f"Could not find or create playlist '{ALL_RECS_PLAYLIST_NAME}' to append songs.")

    print("\nScript finished.")