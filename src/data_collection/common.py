def get_data_directory(game: str) -> str:
    """
    Gets name of the directory where data is stored for a certain game
    """
    return f"./data/{game}/"

def get_length_from_episode_name(episode: str) -> int:
    """
    Get length of an episode from the episode title
    """
    # all files have the format {id)-{length}-{model}.gz
    return int(episode.split("-")[1].split(".")[0])

def get_id_from_episode_name(episode: str) -> int:
    """
    Get id of an episode from the episode title
    """
    # all files have the format {id)-{length}-{model}.gz
    return int(episode.split("-")[0])
