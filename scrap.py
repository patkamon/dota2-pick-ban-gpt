from bs4 import BeautifulSoup
import requests
import pandas as pd

d_pick = []
r_pick = []
d_ban = []
r_ban = []

# d ban1 - r ban1 - d ban2 - r ban2 - d pick1 - r pick1 - r pick2 - d pick2 - d ban3 - r ban3 - d ban4 - r ban4 
# d ban5 - r ban5 - r pick3 - d pick3 - d pick4 - r pick4 - d ban6 - r ban6 - d ban7 - r ban7 - d pick5 - r pick5



# r ban1 - d ban1 - d ban2 - r ban2 - d ban3 - d ban4 - r ban3 - r pick1 - d pick1 - r ban4 - r ban5 - d ban5
# d pick2 - r pick2 - r pick3 - d pick3 - d pick4 - r pick4 - r ban6 - d ban6 - d ban7 - r ban7 - r pick5 - d pick5


# "brkts-popup-body-element-thumbs brkts-popup-body-element-thumbs-right"

url = requests.get("https://liquipedia.net/dota2/Dota_Pro_Circuit/2023/1/North_America/Division_I")
soup = BeautifulSoup(url.content, "html.parser")

print("TITLE: ",(soup.find("title").text).split("-")[0])
tour = (soup.find("title").text).split("-")[0]
# data = soup.find_all("div",{"class":"brkts-popup-body-element brkts-popup-body-game"})



data = soup.find_all("div",{"class":"brkts-popup brkts-match-info-popup"})
merge = pd.DataFrame()


def assign(r_or_d, temp):
    if r_or_d == "r":
        r_pick.append(temp)
    else:
        d_pick.append(temp)

def assign_ban(r_or_d, temp):
    if r_or_d == "r":
        r_ban.append(temp)
    else:
        d_ban.append(temp)


for d in data:
    for ind, div in enumerate(d):
        # get team
        if ind == 0:
            team = div.findChildren("a", recursive=True)
            print(team[0].get("title"), team[-1].get("title"))
            versus = team[0].get("title") + " vs " + team[-1].get("title")
        # get match
        elif ind == 2:
            mat = div.find_all("div", {"class": "brkts-popup-body-element brkts-popup-body-game"})
            print(len(mat))
            for idx, child in enumerate(mat):
                for ii, ch in enumerate(child):
                    if ii == 0:
                        temp = {}
                        # pick hero & side
                        for jj , c in enumerate(ch):
                            if jj == 0:
                                r_or_d = "r" if c.get("class")[0].find("radiant") != -1 else "d" 
                            if jj <5:
                                a = c.findChildren("a" , recursive=False)
                                temp[r_or_d + "_pick"+str(jj+1)] = a[0].get("title")
                        assign(r_or_d, temp)
                    elif ii == len(child)-1:
                        temp = {}
                        # pick hero & side
                        for jj , c in enumerate(ch):
                            if jj == 0:
                                r_or_d = "r" if c.get("class")[0].find("radiant") != -1 else "d" 
                            if jj <5:
                                a = c.findChildren("a" , recursive=False)
                                temp[r_or_d +"_pick"+str(5-jj)] = a[0].get("title")
                        assign(r_or_d, temp)
            ban_div = div.find("div",{"class": "brkts-popup-mapveto"})
            ban_zone = ban_div.find_all("div", {"class": "brkts-popup-body-element-thumbs brkts-popup-body-element-thumbs-right"})
            for idx, ban in enumerate(ban_zone):
                temp = {}
                for ix, b in enumerate(ban):
                    if ix == 0:
                        r_or_d = "r" if b.get("class")[0].find("radiant") != -1 else "d" 
                    a = b.findChildren("a" , recursive=False)
                    temp[r_or_d +"_ban"+str(7-ix)] = a[0].get("title")
                if r_or_d == "r":
                    temp["match"] = versus
                assign_ban(r_or_d, temp)

    df1 = pd.DataFrame(r_pick)
    df2 = pd.DataFrame(d_pick)

    df3 = pd.DataFrame(r_ban)

    df4 = pd.DataFrame(d_ban)


    merge = df1.join(df2)
    merge = merge.join(df3)
    merge = merge.join(df4)

    


merge["tournament"] = tour
# merge.to_csv('pick-ban.csv')
print(merge.head())

df2 = pd.read_csv('pick-ban.csv',index_col=False)

merge.columns = merge.columns.astype(str)
df2.columns = df2.columns.astype(str)

merge = pd.concat([df2,merge], ignore_index=True)
merge = merge.iloc[: , 1:]
merge.to_csv('pick-ban.csv')