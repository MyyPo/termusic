/**
 * model.rs
 * Copyright (C) 2019 gmg137 <gmg137@live.com>
 * Distributed under terms of the GPLv3 license.
 */
use super::super::{ServiceProvider, SongTag};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[allow(unused)]
pub fn to_lyric(json: &str) -> Option<String> {
    if let Ok(value) = serde_json::from_str::<Value>(json) {
        if value.get("code")?.eq(&200) {
            let mut vec: Vec<String> = Vec::new();
            let lyric = value.get("lrc")?.get("lyric")?.as_str()?.to_owned();
            return Some(lyric);
        }
    }
    None
}

// 歌手信息
#[derive(Debug, Deserialize, Serialize)]
pub struct SingerInfo {
    // 歌手 id
    pub id: u64,
    // 歌手姓名
    pub name: String,
    // 歌手照片
    pub pic_url: String,
}

#[allow(unused)]
pub fn to_singer_info(json: &str) -> Option<Vec<SingerInfo>> {
    if let Ok(value) = serde_json::from_str::<Value>(json) {
        if value.get("code")?.eq(&200) {
            let mut vec: Vec<SingerInfo> = Vec::new();
            let array = value.get("result")?.get("artists")?.as_array()?;
            for v in array {
                if let Some(singer_info) = parse_singer_info(v) {
                    vec.push(singer_info);
                }
            }
            return Some(vec);
        }
    }
    None
}

fn parse_singer_info(v: &Value) -> Option<SingerInfo> {
    Some(SingerInfo {
        id: v.get("id")?.as_u64()?,
        name: v.get("name")?.as_str()?.to_owned(),
        pic_url: v
            .get("picUrl")
            .unwrap_or(&json!(""))
            .as_str()
            .unwrap_or("")
            .to_owned(),
    })
}

// 歌曲 URL
#[derive(Debug, Deserialize, Serialize)]
pub struct SongUrl {
    // 歌曲 id
    pub id: u64,
    // 歌曲 URL
    pub url: String,
    // 码率
    pub rate: u64,
}

pub fn to_song_url(json: &str) -> Option<Vec<SongUrl>> {
    if let Ok(value) = serde_json::from_str::<Value>(json) {
        if value.get("code")?.eq(&200) {
            let mut vec: Vec<SongUrl> = Vec::new();
            let array = value.get("data")?.as_array()?;
            for v in array {
                if let Some(url) = parse_song_url(v) {
                    vec.push(url);
                }
            }
            return Some(vec);
        }
    }
    None
}

fn parse_song_url(v: &Value) -> Option<SongUrl> {
    let url = v
        .get("url")
        .unwrap_or(&json!(""))
        .as_str()
        .unwrap_or("")
        .to_owned();
    if !url.is_empty() {
        return Some(SongUrl {
            id: v.get("id")?.as_u64()?,
            url,
            rate: v.get("br")?.as_u64()?,
        });
    }
    None
}

// 歌曲信息
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SongInfo {
    // 歌曲 id
    pub id: u64,
    // 歌名
    pub name: String,
    // 歌手
    pub singer: String,
    // 专辑
    pub album: String,
    // 封面图
    pub pic_url: String,
    // 歌曲时长
    pub duration: String,
    // 歌曲链接
    pub song_url: String,
}

// parse: 解析方式
pub fn to_song_info(json: &str, parse: Parse) -> Option<Vec<SongTag>> {
    if let Ok(value) = serde_json::from_str::<Value>(json) {
        if value.get("code")?.eq(&200) {
            let mut vec: Vec<SongTag> = Vec::new();
            if let Parse::Search = parse {
                let array = value.get("result")?.as_object()?.get("songs")?.as_array()?;
                for v in array {
                    if let Some(item) = parse_song_info(v) {
                        vec.push(item);
                    }
                }
            }
            return Some(vec);
        }
    }
    None
}

fn parse_song_info(v: &Value) -> Option<SongTag> {
    // let _duration = v.get("duration")?.as_u64()?;
    Some(SongTag {
        artist: Some(
            v.get("artists")?
                .get(0)?
                .get("name")
                .unwrap_or(&json!("Unknown Artist"))
                .as_str()
                .unwrap_or("Unknown Artist")
                .to_owned(),
        ),
        title: Some(v.get("name")?.as_str()?.to_owned()),
        album: Some(
            v.get("album")?
                .get("name")
                .unwrap_or(&json!("Unknown Album"))
                .as_str()
                .unwrap_or("Unknown Album")
                .to_owned(),
        ),
        lang_ext: Some(String::from("netease")),
        lyric_id: Some(v.get("id")?.as_u64()?.to_string()),
        song_id: Some(v.get("id")?.as_u64()?.to_string()),
        service_provider: Some(ServiceProvider::Netease),
        url: Some(
            if v.get("fee")?.as_u64()? == 0 {
                "Downloadable"
            } else {
                "Copyright protected"
            }
            .to_string(),
        ),
        pic_id: Some(
            v.get("album")?
                .get("picId")
                .unwrap_or(&json!("Unknown"))
                .as_u64()?
                .to_string(),
        ),
        album_id: Some(
            v.get("album")?
                .get("picId")
                .unwrap_or(&json!("Unknown"))
                .as_u64()?
                .to_string(),
        ),
    })
}

// 歌单信息
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SongList {
    // 歌单 id
    pub id: u64,
    // 歌单名
    pub name: String,
    // 歌单封面
    pub cover_img_url: String,
}

// 登陆信息
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoginInfo {
    // 登陆状态码
    pub code: i32,
    // 用户 id
    pub uid: u64,
    // 用户昵称
    pub nickname: String,
    // 用户头像
    pub avatar_url: String,
    // 状态消息
    pub msg: String,
}

// 请求方式
#[allow(unused)]
#[derive(Clone, Copy, Debug)]
pub enum Method {
    Post,
    Get,
}

// 解析方式
// USL: 用户
// UCD: 云盘
// RMD: 推荐
// RMDS: 推荐歌曲
// SEARCH: 搜索
// SD: 单曲详情
// ALBUM: 专辑
// TOP: 热门
#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub enum Parse {
    Search,
    Usl,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_parse_songinfo() {
        let sample_data = r#"{
            "result": {
              "songs": [
                {
                  "id": 1000000000,
                  "name": "Track A",
                  "artists": [
                    {
                      "id": 3333333,
                      "name": "Some Artist",
                      "picUrl": null,
                      "alias": [],
                      "albumSize": 0,
                      "picId": 0,
                      "fansGroup": null,
                      "img1v1Url": "https://p3.music.126.net/AAAAAAAAAAAAAAAAAAAAAAAA/0000000000000000.jpg",
                      "img1v1": 0,
                      "trans": null
                    }
                  ],
                  "album": {
                    "id": 100000001,
                    "name": "Some Album 1",
                    "artist": {
                      "id": 0,
                      "name": "",
                      "picUrl": null,
                      "alias": [],
                      "albumSize": 0,
                      "picId": 0,
                      "fansGroup": null,
                      "img1v1Url": "https://p3.music.126.net/AAAAAAAAAAAAAAAAAAAAAAAA/0000000000000000.jpg",
                      "img1v1": 0,
                      "trans": null
                    },
                    "publishTime": 1111111111111,
                    "size": 6,
                    "copyrightId": 1111,
                    "status": 1,
                    "picId": 444444444444444444,
                    "mark": 0
                  },
                  "duration": 89595,
                  "copyrightId": 1111,
                  "status": 0,
                  "alias": [],
                  "rtype": 0,
                  "ftype": 0,
                  "mvid": 0,
                  "fee": 1,
                  "rUrl": null,
                  "mark": 66666666666
                },
                {
                  "id": 1111111111,
                  "name": "Track B",
                  "artists": [
                    {
                      "id": 3333333,
                      "name": "Some Artist",
                      "picUrl": null,
                      "alias": [],
                      "albumSize": 0,
                      "picId": 0,
                      "fansGroup": null,
                      "img1v1Url": "https://p4.music.126.net/AAAAAAAAAAAAAAAAAAAAAAAA/0000000000000000.jpg",
                      "img1v1": 0,
                      "trans": null
                    }
                  ],
                  "album": {
                    "id": 11111112,
                    "name": "Some Album 2",
                    "artist": {
                      "id": 0,
                      "name": "",
                      "picUrl": null,
                      "alias": [],
                      "albumSize": 0,
                      "picId": 0,
                      "fansGroup": null,
                      "img1v1Url": "https://p3.music.126.net/AAAAAAAAAAAAAAAAAAAAAAAA/0000000000000000.jpg",
                      "img1v1": 0,
                      "trans": null
                    },
                    "publishTime": 2222222222222,
                    "size": 4,
                    "copyrightId": 1111,
                    "status": 1,
                    "picId": 555555555555555555,
                    "mark": 0
                  },
                  "duration": 158143,
                  "copyrightId": 1111,
                  "status": 0,
                  "alias": [],
                  "rtype": 0,
                  "ftype": 0,
                  "mvid": 8888888,
                  "fee": 1,
                  "rUrl": null,
                  "mark": 77777777777
                }
              ],
              "hasMore": false,
              "songCount": 20
            },
            "code": 200
          }"#;

        let res = to_song_info(sample_data, Parse::Search).unwrap();

        assert_eq!(res.len(), 2);

        const ARTIST: &str = "Some Artist";

        assert_eq!(
            res[0],
            SongTag {
                artist: Some(ARTIST.to_owned()),
                title: Some("Track A".to_owned()),
                album: Some("Some Album 1".to_owned()),
                lang_ext: Some("netease".to_string()),
                service_provider: Some(ServiceProvider::Netease),
                song_id: Some("1000000000".to_owned()),
                lyric_id: Some("1000000000".to_owned()),
                url: Some("Copyright protected".to_owned()),
                pic_id: Some("444444444444444444".to_owned()),
                album_id: Some("444444444444444444".to_owned())
            }
        );

        assert_eq!(
            res[1],
            SongTag {
                artist: Some(ARTIST.to_owned()),
                title: Some("Track B".to_owned()),
                album: Some("Some Album 2".to_owned()),
                lang_ext: Some("netease".to_string()),
                service_provider: Some(ServiceProvider::Netease),
                song_id: Some("1111111111".to_owned()),
                lyric_id: Some("1111111111".to_owned()),
                url: Some("Copyright protected".to_owned()),
                pic_id: Some("555555555555555555".to_owned()),
                album_id: Some("555555555555555555".to_owned())
            }
        );
    }
}
