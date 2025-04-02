# import SoccerNet
# from SoccerNet.Downloader import SoccerNetDownloader
# mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="/home/xavier/Projects/MasterCV/C6_Lab/Week5/dataset"
# )

# # Broadcast Videos
# mySoccerNetDownloader.password = ""
# mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test","challenge"])
# mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train","valid","test","challenge"])

# # Video features at 2 frames per second
# mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["train","valid","test","challenge"])
# mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split=["train","valid","test","challenge"])

# # Action and replay images
# mySoccerNetDownloader.downloadGames(files=["Frames-v3.zip"], split=["train","valid","test"], task="frames")

# # Action spotting labels
# mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])

# # Replay grounding labels
# mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["train","valid","test"])

# # Calibration data and abels
# mySoccerNetDownloader.downloadDataTask(task="calibration", split=["train","valid","test","challenge"]) 
# mySoccerNetDownloader.downloadDataTask(task="calibration-2023", split=["train","valid","test","challenge"])

# # Re-identification and labels
# mySoccerNetDownloader.downloadDataTask(task="reid", split=["train", "valid", "test", "challenge"])
# mySoccerNetDownloader.downloadDataTask(task="reid-2023", split=["train", "valid", "test", "challenge"])

# # Tracking data and labels
# mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])
# mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])

# # Jersey number and data labels
# mySoccerNetDownloader.downloadDataTask(task="jersey-2023", split=["train","test","challenge"])

# # Dense Video Captioning labels
# mySoccerNetDownloader.downloadDataTask(task="caption-2023", split=["train","valid", "test","challenge"])

# # Ball action data and labels
# mySoccerNetDownloader.downloadDataTask(task="spotting-ball-2023", split=["train", "valid", "test", "challenge"], password="") 


from huggingface_hub import snapshot_download
snapshot_download(repo_id="SoccerNet/SN-BAS-2025",
                  repo_type="dataset", revision="main",
                  local_dir="SoccerNet/SN-BAS-2025")
