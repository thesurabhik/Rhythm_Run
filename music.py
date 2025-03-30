import time
import threading
import pygame

# Define constants for note heights
LOW = "LOW"
MID = "MID"
HIGH = "HIGH"
NONE = "NONE"
DONE = "DONE"  # Marker for end of song

# Hardcoded song with time offsets in seconds
song = [
    {"height": NONE, "time_offset": 0},  # No note at start
    {"height": LOW, "time_offset": 6},  # Note at 6s
    {"height": MID, "time_offset": 8},  # Note at 8s
    {"height": HIGH, "time_offset": 10},  # Note at 10
    {"height": LOW, "time_offset": 12},  # Note at 12
    {"height": MID, "time_offset": 14},  # Note at 14
    {"height": HIGH, "time_offset": 16},  # Note at 16
    {"height": LOW, "time_offset": 18},  # Note at 18
    {"height": HIGH, "time_offset": 20},  # Note at 20
]

# Start time
start_time = time.time()

# Play a song at the start in a separate thread
def play_start_song():
    print("Playing start song...")
    pygame.mixer.init()
    pygame.mixer.music.load("/Users/surabhi/Documents/college/s25/hackathon/Rhythm_Run/TeenageDream.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass

# Function to process notes at the correct time
def test_notes():
    # Start the music in a separate thread
    music_thread = threading.Thread(target=play_start_song)
    music_thread.start()

    for note in song:
        note_time = note["time_offset"]  # Get note's time offset in seconds
        time_to_wait = note_time - (time.time() - start_time)

        # Wait until it's time to play the note
        if time_to_wait > 0:
            time.sleep(time_to_wait)

        # Get the actual elapsed time
        current_time = time.time() - start_time

        # Stop when the song is done
        if note["height"] == DONE:
            print(f"{current_time:.2f}s - Test completed.")
            break
        else:
            print(f"{current_time:.2f}s - {note['height']}")

    # Stop the music after processing all notes
    pygame.mixer.music.stop()

    # Wait for the music thread to finish
    music_thread.join()

# Run the test
test_notes()
