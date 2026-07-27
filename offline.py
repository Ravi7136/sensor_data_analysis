COLD_ROOM_WAPS = ["abc-2A-ap01", "abc-2B-ap02", "abc-4A-ap03"]


def get_offline_exit_eventtime(df):
    """Return the EVENTTIME of the first OFFLINE_WITH_LOCATION event in the
    final trailing block of offline events from the cold room WAPs.

    Offline events that are followed by a later online event (e.g. a
    CHECKPOINT_WITH_LOCATION) are ignored, since the package came back
    online afterwards. Returns None if the readings do not end with an
    OFFLINE_WITH_LOCATION event."""
    cold_room = df[df["HARDWARENAME"].isin(COLD_ROOM_WAPS)].sort_values("EVENTTIME")
    if cold_room.empty:
        return None

    is_offline = (cold_room["BLE_DESCRIPTION"] == "OFFLINE_WITH_LOCATION").to_numpy()
    if not is_offline[-1]:
        return None  # readings do not end with an offline event

    online_positions = (~is_offline).nonzero()[0]
    block_start = online_positions[-1] + 1 if len(online_positions) else 0
    return cold_room.iloc[block_start]["EVENTTIME"]


offline_exit_eventtime = get_offline_exit_eventtime(df)
print(offline_exit_eventtime)
