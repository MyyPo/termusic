#[cfg(all(feature = "gst", not(feature = "mpv")))]
mod gstreamer_backend;
#[cfg(feature = "mpv")]
mod mpv_backend;
mod playlist;
#[cfg(not(any(feature = "mpv", feature = "gst")))]
mod rusty_backend;
use crate::config::Termusic;
use anyhow::Result;
#[cfg(feature = "mpv")]
use mpv_backend::Mpv;
pub use playlist::Playlist;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::Receiver;

#[derive(Clone, Copy, PartialEq)]
pub enum Status {
    Running,
    Stopped,
    Paused,
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Running => write!(f, "Running"),
            Self::Stopped => write!(f, "Stopped"),
            Self::Paused => write!(f, "Paused"),
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub enum Loop {
    Single,
    Playlist,
    Queue,
}

#[allow(clippy::non_ascii_literal)]
impl Loop {
    pub fn display(&self, display_symbol: bool) -> String {
        if display_symbol {
            match self {
                Self::Single => "🔂".to_string(),
                Self::Playlist => "🔁".to_string(),
                Self::Queue => "⬇".to_string(),
            }
        } else {
            match self {
                Self::Single => "single".to_string(),
                Self::Playlist => "playlist".to_string(),
                Self::Queue => "consume".to_string(),
            }
        }
    }
}

pub enum PlayerMsg {
    AboutToFinish,
}

pub struct GeneralPlayer {
    #[cfg(all(feature = "gst", not(feature = "mpv")))]
    player: gstreamer_backend::GStreamer,
    #[cfg(feature = "mpv")]
    player: Mpv,
    #[cfg(not(any(feature = "mpv", feature = "gst")))]
    player: rusty_backend::Player,
    pub message_rx: Receiver<PlayerMsg>,
    pub playlist: Playlist,
    status: Status,
    config: Termusic,
}

impl GeneralPlayer {
    pub fn new(config: &Termusic) -> Self {
        #[cfg(all(feature = "gst", not(feature = "mpv")))]
        let player = gstreamer_backend::GStreamer::new(config);
        #[cfg(feature = "mpv")]
        let player = Mpv::new(config);
        #[cfg(not(any(feature = "mpv", feature = "gst")))]
        let (player, message_rx) = rusty_backend::Player::new(config);
        let mut playlist = Playlist::default();
        if let Ok(p) = Playlist::new() {
            playlist = p;
        }
        Self {
            player,
            message_rx,
            playlist,
            status: Status::Stopped,
            config: config.clone(),
        }
    }
    pub fn toggle_gapless(&mut self) {
        self.player.gapless = !self.player.gapless;
    }

    pub fn next(&mut self) {
        if let Some(song) = self.playlist.tracks.pop_front() {
            if let Some(file) = song.file() {
                self.add_and_play(file);
            }
            match self.config.loop_mode {
                Loop::Playlist => self.playlist.tracks.push_back(song.clone()),
                Loop::Single => self.playlist.tracks.push_front(song.clone()),
                Loop::Queue => {}
            }
            self.playlist.current_track = Some(song);
        }
    }

    // pub fn start_play(&mut self) {
    //     for track in self
    //         .playlist
    //         .as_slice()
    //         .iter()
    //         .filter_map(|track| track.file())
    //     {
    //         self.player.enqueue(track);
    //     }
    // }

    // pub fn run(&mut self) {
    //     match self.status {
    //         Status::Stopped => {
    //             self.start_play();
    //         }
    //         Status::Running | Status::Paused => {}
    //     }
    // }

    pub fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    pub fn is_stopped(&self) -> bool {
        self.status == Status::Stopped
    }

    pub fn status(&self) -> Status {
        self.status
    }
}

impl GeneralP for GeneralPlayer {
    fn add_and_play(&mut self, current_track: &str) {
        self.player.add_and_play(current_track);
    }
    fn volume(&self) -> i32 {
        self.player.volume()
    }
    fn volume_up(&mut self) {
        self.player.volume_up();
    }
    fn volume_down(&mut self) {
        self.player.volume_down();
    }
    fn set_volume(&mut self, volume: i32) {
        self.player.set_volume(volume);
    }
    fn pause(&mut self) {
        self.player.pause();
    }
    fn resume(&mut self) {
        self.player.resume();
    }
    fn is_paused(&self) -> bool {
        self.player.is_paused()
    }
    fn seek(&mut self, secs: i64) -> Result<()> {
        self.player.seek(secs)
    }
    fn get_progress(&mut self) -> Result<(f64, i64, i64)> {
        self.player.get_progress()
    }

    fn set_speed(&mut self, speed: f32) {
        self.player.set_speed(speed);
    }

    fn speed_up(&mut self) {
        self.player.speed_up();
    }

    fn speed_down(&mut self) {
        self.player.speed_down();
    }

    fn speed(&self) -> f32 {
        self.player.speed()
    }

    fn stop(&mut self) {
        self.player.stop();
    }
}

pub trait GeneralP {
    // fn start_play(&mut self);
    fn add_and_play(&mut self, current_track: &str);
    fn volume(&self) -> i32;
    fn volume_up(&mut self);
    fn volume_down(&mut self);
    fn set_volume(&mut self, volume: i32);
    fn pause(&mut self);
    fn resume(&mut self);
    fn is_paused(&self) -> bool;
    fn seek(&mut self, secs: i64) -> Result<()>;
    fn get_progress(&mut self) -> Result<(f64, i64, i64)>;
    fn set_speed(&mut self, speed: f32);
    fn speed_up(&mut self);
    fn speed_down(&mut self);
    fn speed(&self) -> f32;
    fn stop(&mut self);
}
