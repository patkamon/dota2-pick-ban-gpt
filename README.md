# dota2-pick-ban-gpt   
train gpt on sequence of pick/ban hero in dota2.      
Based on this [video](https://www.youtube.com/watch?v=kCc8FmEb1nY).

<img src='https://github.com/patkamon/dota2-pick-ban-gpt/blob/main/web.png' />

I also upload dataset on [kaggle](https://www.kaggle.com/search?q=pick+ban+dota2+sortBy%3Adate)

# File


- `scrap.py` scrap from liquipedia
- `pick-ban.csv` df in csv
- `write-txt.ipynb` parse csv(df) to txt and format
- `pick-ban` dataset
- `pred.ipynb` train model and save
- `./model/model-20000` model
- `./model/token-encode-20000.json` token encoder
- `./model/token-decode-20000.json` token decoder
- `load.ipynb` load model and generate output (pick ban hero)
- `main.py` fastapi for rest framework
- `pick-ban-front` nextjs app for frontend web visualise



  # to train yourself
  run `pred.ipynb`

  # to visualize
  use this two file
- `main.py` fastapi for rest framework
    - uvicorn main:app 
- `pick-ban-front` nextjs app for frontend web visualise
    - run `npm run dev` make sure you have npm 18 (`nvm use 18`)
