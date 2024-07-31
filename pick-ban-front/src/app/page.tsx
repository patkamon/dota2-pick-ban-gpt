/* eslint-disable jsx-a11y/alt-text */
/* eslint-disable @next/next/no-img-element */
"use client";

import { useState } from "react";
import hero from "../../hero-img.json";
import base from "../../hero.json";


export default function Home() {
  // const [radiantPick, setRadiantPick] = useState([1, 2, 3, 4, 5]);
  // const [direPick, setDirePick] = useState([1, 2, 3, 4, 5]);

  // const [radiantBan, setRadiantBan] = useState([1, 2, 3, 4, 5, 6, 7]);
  // const [direBan, setDireBan] = useState([1, 2, 3, 4, 5, 6, 7]);

  const prefix_url = "https://cdn.dota2.com/apps/dota2/images/heroes/";
  const suffix_url = "_lg.png";

  const [index, setIndex] = useState(0);

  const [pred, setPred] = useState([
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
  ] as undefined[] | object[]);

  const [list, setList] = useState([
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
  ] as undefined[] | string[]);

  // r ban1 - d ban1 - d ban2 - r ban2 - d ban3 - d ban4 - r ban3 - r pick1 - d pick1 - r ban4 - r ban5 - d ban5
  // d pick2 - r pick2 - r pick3 - d pick3 - d pick4 - r pick4 - r ban6 - d ban6 - d ban7 - r ban7 - r pick5 - d pick5

  function filt(val: string | undefined) {
    let copyList = list;
    let i = 0;
    let newArray = copyList.filter((value) => {
      if (value == val) {
        i += 1;
      }
      return value !== val;
    }) as string[] | undefined[];

    for (let x = 0; x < i; x++) {
      setList((copyList) => [...copyList, undefined] as string[] | undefined[]);
    }

    console.log(newArray, index - i);
    console.log(index, i);
    setIndex(index - i);
    setList(newArray);
  }

  function getKeyByValue(object: any, value: string) {
    return Object.keys(object).find(key => object[key] === value);
  }


  function prep(){
    const l = list.filter((d)=>{
      return d != undefined
    })

    const newdata = l.map((d: any)=>{
      let k = getKeyByValue(hero, d)
      return base[k as keyof typeof base]
    })

    console.log(newdata)
    console.log(newdata.join("/") + "/")
    
    return newdata.join("/") + "/"
  }




  async function prediction(){
    const pre = prep()

    const resp = await fetch("http://127.0.0.1:8000/predict?crypt="+pre)
    const data = await resp.json()
    console.log(data)

    const newdata = data.pred.map((d: any)=>{
      let k = getKeyByValue(base, d.hero)
      return {'prob': d.prob, 'hero': hero[k as keyof typeof hero]}
    })

    console.log("newdata",newdata)


    setPred(newdata)
  }

  return (
    <main className="flex min-h-screen flex-col items-center gap-8 bg-gray-700 p-14">
      <div>
        <div className="flex gap-4">
          {pred.map((p, ind) => {
            return (
              <div key={"radiantpick" + ind} className="flex flex-col items-center justify-center">
                <div
                  onClick={() => {
                    if (
                      p != undefined &&
                      !list.includes(
                        ((p as any).hero) as never
                      )
                    ) {
                      let copyList = list;
                      copyList[index] = (p as any).hero
                      console.log(copyList);
                      let ind = index;
                      ind += 1;
                      setIndex(ind);
                      setList(copyList);
                      console.log(index);
                    }
                  }}
                  className="w-24 h-24 border-2 bg-yellow-800 text-white"
                >
                  {p ? ((p as any).prob*100).toFixed(2) + "%" : ind}
                  <img
                    src={
                      p != undefined && ! list.includes(
                        (p as any).hero as never
                      )
                        ? prefix_url + (p as any).hero.toLowerCase().replace(" ", "_") + suffix_url
                        : "./logo.png"
                    }
                  ></img>
                </div>
                <h1 className="mb-2 text-white">{p ? (p as any).hero : 'hero name'}</h1>
              </div>
            );
          })}
        </div>
      </div>

      <div className="flex justify-between gap-24 items-center">
        <div className="flex">
          {[list[7], list[13], list[14], list[17], list[22]].map((p, ind) => {
            return (
              <div
                onClick={() => filt(p)}
                className="w-24 h-24 border-2 bg-green-900 text-white"
                key={"radiantpick" + ind}
              >
                {ind + 1}
                <img
                  src={
                    p != undefined ? prefix_url + p.toLowerCase().replace(" ", "_") + suffix_url : "./logo.png"
                  }
                ></img>
              </div>
            );
          })}
        </div>
        <button onClick={()=>prediction()} className="border-2 border-black bg-green-400 px-6 py-2">
          Predict
        </button>

        <div className="flex">
          {[list[8], list[12], list[15], list[16], list[23]].map((p, ind) => {
            return (
              <div
                onClick={() => filt(p)}
                className="w-24 h-24 border-2 bg-red-900 text-white"
                key={"direpick" + ind}
              >
                {ind + 1}
                <img
                  src={
                    p != undefined ? prefix_url + p.toLowerCase().replace(" ", "_") + suffix_url : "./logo.png"
                  }
                ></img>
              </div>
            );
          })}
        </div>
      </div>

      <div className="flex justify-between gap-12">
        <div className="flex">
          {[
            list[0],
            list[3],
            list[6],
            list[9],
            list[10],
            list[18],
            list[21],
          ].map((p, ind) => {
            return (
              <div
                onClick={() => filt(p)}
                className="w-24 h-24 border-2 bg-green-900 text-white"
                key={"radiantban" + ind}
              >
                {ind + 1}
                <img
                  src={
                    p != undefined ? prefix_url + p.toLowerCase().replace(" ", "_") + suffix_url : "./logo.png"
                  }
                ></img>
              </div>
            );
          })}
        </div>

        <div className="flex">
          {[
            list[1],
            list[2],
            list[4],
            list[5],
            list[11],
            list[19],
            list[20],
          ].map((p, ind) => {
            return (
              <div
                onClick={() => filt(p)}
                className="w-24 h-24 border-2 bg-red-900 text-white"
                key={"direban" + ind}
              >
                {ind + 1}
                <img
                  src={
                    p != undefined ? prefix_url + p.toLowerCase().replace(" ", "_") + suffix_url : "./logo.png"
                  }
                ></img>
              </div>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-10 gap-4">
        {Object.keys(hero).map((x, ind) => {
          return (
            <div
              key={x + ind}
              onClick={() => {
                if (
                  !list.includes(
                    hero[x as keyof typeof hero] as never
                      // .toLowerCase()
                      // .replace(" ", "_") 
                  )
                ) {
                  let copyList = list;
                  copyList[index] = hero[x as keyof typeof hero]
                    // .toLowerCase()
                    // .replace(" ", "_");
                  console.log(copyList);
                  let ind = index;
                  ind += 1;
                  setIndex(ind);
                  setList(copyList);
                  console.log(index);
                }
              }}
            >
              <h1 className="text-white text-xs text-clip">
                {hero[x as keyof typeof hero].toLowerCase()}
              </h1>
              <img
                src={
                  list.includes(
                    hero[x as keyof typeof hero] as never
                  ) ? "./logo.png":
                  prefix_url +
                  hero[x as keyof typeof hero].toLowerCase().replace(" ", "_") +
                  suffix_url
                }
                alt={hero[x as keyof typeof hero]
                  .toLowerCase()
                  .replace(" ", "_")}
              ></img>
            </div>
          );
        })}
      </div>
    </main>
  );
}
