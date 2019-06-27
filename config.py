"""
configuration stuff
"""

DB_FILENAME = 'datastore/swagmaster-with-intervals.db'

HOTFILTER = "WHERE song_hotttnesss>0 AND artist_hotttnesss>0 AND beats_avg>0 AND bars_avg>0"

TRAIN_FRAC = 0.60
CV_FRAC = 0.20
TEST_FRAC = 1 - TRAIN_FRAC - CV_FRAC


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
INDEX_segments_avg        = 21
INDEX_tatums_avg          = 22
INDEX_beats_avg           = 23
INDEX_bars_avg            = 24
INDEX_sections_avg        = 25


# # Acoustic Features 
#     + key
#     + tempo 
#     + loudness 
#     + time_signature 
#     + tatums_avg 
#     + segments_avg
#     + beats_avg 
#     + bars_avg 
#     + sections_avg 
acoustic_features = """
                    key tempo loudness time_signature tatums_avg segments_avg 
                    beats_avg bars_avg sections_avg
                    """.split()




# # Metadata Features 
#   + duration
#   + artist_familiartiy 
#   + artist_hotttnesss 
metadata_feaures = "duration artist_hotttnesss artist_familiarity".split()  


# # New features 
#   + energy 
#   + danceability 
constructed_features = "energy danceability".split()  
energy_features = "energy1 energy2 energy3 energy4".split() 
dance_features = "dance1 dance2 dance3 dance4".split() 
