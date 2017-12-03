"""
configuration stuff
"""

DB_FILENAME = 'swagmaster.db'

HOTFILTER = "WHERE song_hotttnesss>0 AND artist_hotttnesss>0"

TRAIN_FRAC = 0.75
TEST_FRAC = 1-TRAIN_FRAC


INDEX_track_id            = 0
INDEX_title               = 1
INDEX_song_id             = 2
INDEX_release             = 3
INDEX_artist_id           = 4
INDEX_artist_mbid         = 5
INDEX_artist_name         = 6
INDEX_duration            = 7
INDEX_artist_familiarity  = 8
INDEX_artist_hotttnesss   = 9
INDEX_year                = 10
INDEX_track_7digitalid    = 11
INDEX_shs_perf            = 12
INDEX_shs_work            = 13
INDEX_song_hotttnesss     = 14
INDEX_danceability        = 15 
INDEX_energy              = 16
INDEX_key                 = 17
INDEX_tempo               = 18
INDEX_loudness            = 19
INDEX_time_signature      = 20


