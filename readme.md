# AI 530 MSD project

## Notes

### Tutorial notebooks 
[MSD link to tutorials](https://labrosa.ee.columbia.edu/millionsong/pages/tutorial)
#### tutorial_1
+ Shows how to iterate over the files within the MillionSongSubset
+ The AdditionFiles has sql databases set up to ping into the /data folder's contents 
+ Runs through an exercise to find out which artist has the most songs in the dataset (by artist_id)


#### tutorial_3_track-metadata 
+ Shows how to interface with the dataset (in db form) using sqlite.
    * There are .db files in AdditionalFiles.  This one uses track_metadata (subset_track_metadata.db)

+ `subset_track_metadata.db`
    * Contains one table named 'songs'
    * Contains the following columns
        - track_id text PRIMARY KEY, 
        - title text, 
        - song_id text, 
        - release text, 
        - artist_id text, 
        - artist_mbid text, 
        - artist_name text, 
        - duration real, 
        - artist_familiarity real, 
        - artist_hotttnesss real, 
        - year int
+ Some useful queries:
    * Get all songs without MB ID's : SELECT artist_id,artist_mbid FROM songs WHERE artist_mbid=''
    * Get all distinct artists: SELECT DISTINCT artist_id, artist_name FROM songs
    * Get all dudes with a float>value: SELECT DISTINCT artist_name, artist_familiarity FROM songs WHERE artist_familiarity>.8
        - Can use this one to filter out the tracks where hotttnesss is 0. (empty data) (WHERE NOT artist_hotttnesss=0)

### Potentially useful links

[github link -- matches artist names from yahoo ratings set to the MSD (by artist_id)](https://github.com/tbertinmahieux/MSongsDB/blob/master/Tasks_Demos/YahooRatings/match_artist_names.py)

[gitlub link -- MSongDB code.  Has python scripts and examples and stuff](https://github.com/tbertinmahieux/MSongsDB)

[github link -- homeboi kevin over here did all the tutorials in python notebooks](https://github.com/kevin11hg/msong)

[google group forum thing](https://groups.google.com/forum/#!forum/millionsongdataset)