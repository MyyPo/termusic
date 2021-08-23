use super::super::{SongTag, SongtagProvider};
/**
 * MIT License
 *
 * termusic - Copyright (c) 2021 Larry Hao
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
use super::NCMResult;
use custom_error::custom_error;
use serde_json::{json, Value};

#[allow(unused)]
pub fn to_lyric(json: String) -> NCMResult<String> {
    let value = serde_json::from_str::<Value>(&json)?;
    if value.get("msg").ok_or(Errors::NoneError)?.eq("成功") {
        let lyric = value
            .get("lyric")
            .ok_or(Errors::NoneError)?
            .as_str()
            .ok_or(Errors::NoneError)?
            .to_owned();
        return Ok(lyric);
    }
    Err(Errors::NoneError)
}

#[allow(unused)]
pub fn to_pic_url(json: String) -> NCMResult<String> {
    let value = serde_json::from_str::<Value>(&json)?;
    if value.get("msg").ok_or(Errors::NoneError)?.eq("成功") {
        let mut vec: Vec<String> = Vec::new();
        let pic_url = value
            .get("largePic")
            .ok_or(Errors::NoneError)?
            .as_str()
            .ok_or(Errors::NoneError)?
            .to_owned();
        return Ok(pic_url);
    }
    Err(Errors::NoneError)
}

pub fn to_song_info(json: String, parse: Parse) -> NCMResult<Vec<SongTag>> {
    let value = serde_json::from_str::<Value>(&json)?;
    if value.get("success").ok_or(Errors::NoneError)?.eq(&true) {
        let mut vec: Vec<SongTag> = Vec::new();
        let list = json!([]);
        if let Parse::SEARCH = parse {
            let array = value
                .get("musics")
                .unwrap_or(&list)
                .as_array()
                .ok_or(Errors::NoneError)?;
            for v in array.iter() {
                let pic_id = v
                    .get("cover")
                    .unwrap_or(&json!("N/A"))
                    .as_str()
                    .unwrap_or("")
                    .to_owned();
                let artist = v
                    .get("singerName")
                    .unwrap_or(&json!("未知"))
                    .as_str()
                    .unwrap_or("未知")
                    .to_owned();
                let title = v
                    .get("songName")
                    .ok_or(Errors::NoneError)?
                    .as_str()
                    .ok_or(Errors::NoneError)?
                    .to_owned();

                let album_id = v
                    .get("albumId")
                    .ok_or(Errors::NoneError)?
                    .as_str()
                    .ok_or(Errors::NoneError)?
                    .to_owned();

                let url = v
                    .get("mp3")
                    .unwrap_or(&json!("N/A"))
                    .as_str()
                    .unwrap_or("Copyright protected")
                    .to_owned();

                vec.push(SongTag {
                    song_id: Some(
                        v.get("id")
                            .ok_or(Errors::NoneError)?
                            .as_str()
                            .ok_or(Errors::NoneError)?
                            .to_owned(),
                    ),
                    title: Some(title),
                    artist: Some(artist),
                    album: Some(
                        v.get("albumName")
                            .unwrap_or(&json!("未知"))
                            .as_str()
                            .unwrap_or("")
                            .to_owned(),
                    ),
                    pic_id: Some(pic_id),
                    lang_ext: Some("chi".to_string()),
                    service_provider: Some(SongtagProvider::Migu),
                    lyric_id: Some(
                        v.get("copyrightId")
                            .ok_or(Errors::NoneError)?
                            .as_str()
                            .ok_or(Errors::NoneError)?
                            .to_owned(),
                    ),
                    url: Some(url),
                    album_id: Some(album_id),
                });
            }
            return Ok(vec);
        }
    }
    Err(Errors::NoneError)
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
#[allow(unused, clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub enum Parse {
    USL,
    UCD,
    RMD,
    RMDS,
    SEARCH,
    SD,
    ALBUM,
    TOP,
}

custom_error! { pub Errors
    OpenSSL{ source: openssl::error::ErrorStack } = "openSSL Error",
    Regex{ source: regex::Error } = "regex Error",
    SerdeJson{ source: serde_json::error::Error } = "serde json Error",
    Parse{ source: std::num::ParseIntError } = "parse Error",
    // AsyncIoError{ source: io::Error } = "async io Error",
    // IsahcError{ source: isahc::Error } = "isahc Error",
    NoneError = "None Error",
    // FromUtf8Error{source: std::string::FromUtf8Error} = "UTF8 Error",
}
