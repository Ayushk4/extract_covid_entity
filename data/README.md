# Data

**Update: Dec 2020**: The dataset is no longer public due to Twitter Privacy Policy. To get access to the dataset, please mail `zong.56@osu.edu` and cc `alan.ritter@cc.gatech.edu`. 

Obtain annotation and tweet id files for 5 events  from [this link](https://github.com/viczong/extract_COVID19_events_from_Twitter/tree/master/data)

Then follow [these instructions](https://github.com/viczong/extract_COVID19_events_from_Twitter#download-tweets-and-preprocessing) for downloading and data preprocessing.

0. Prepare your Twitter API keys and tokens from [developer.twitter.com](developer.twitter.com) . You will have to fill up a form telling why you need the tweets. Typically the authentication doesn't take more than a couple of days, if you are using it for research purposes. We cannot share their tweets as twitter doesn't allow it.
1. Download tweets using `python download_data.py --API_key your_API_key --API_secret_key your_API_secret_key --access_token your_access_token --access_token_secret your_access_token_secret`. Execute the above command from the parent directory of this folder where `download_data.py` is present.
2. Tweet parsing/tokenizing/: Follow the instructions from [this section here](https://github.com/viczong/extract_COVID19_events_from_Twitter#tweets-parsing-and-pre-processing) till before Models training and results section.

The [shared task organizers](http://noisy-text.github.io/2020/extract_covid19_event-shared_task.html) are yet to publicly release the test data. 
