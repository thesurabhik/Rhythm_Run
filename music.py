import time

# Define constants for note heights
LOW = "LOW"
MID = "MID"
HIGH = "HIGH"
NONE = "NONE"
DONE = "DONE"  # Marker for end of song

# Hardcoded song with time offsets in seconds
song = [
    {"height": NONE, "time_offset": 0},  # No note at 3s
    {"height": LOW, "time_offset": 6},  # Note at 5s
    {"height": MID, "time_offset": 8},  # No note at 7s
    {"height": HIGH, "time_offset": 10},  # Note at 9s
    {"height": LOW, "time_offset": 12},  # No note at 12s
    {"height": MID, "time_offset": 14},  # Note at 15s
    {"height": HIGH, "time_offset": 16},  # Note at 18s
    {"height": LOW, "time_offset": 18},  # End of song at 20s
    {"height": HIGH, "time_offset": 20},  # End of song at 20s
]

# Start time
start_time = time.time()

# Function to process notes at the correct time
def test_notes():
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

# Run the test
test_notes()
