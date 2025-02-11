/*
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

// #![forbid(unsafe_code)]
#![recursion_limit = "2048"]
#![warn(clippy::all, clippy::correctness)]
#![warn(rust_2018_idioms)]
#![warn(clippy::pedantic)]
#[allow(clippy::pedantic)]
pub mod player {
    tonic::include_proto!("player");

    // implement transform function for easy use
    impl From<Duration> for std::time::Duration {
        fn from(value: Duration) -> Self {
            std::time::Duration::new(value.secs, value.nanos)
        }
    }

    impl From<std::time::Duration> for Duration {
        fn from(value: std::time::Duration) -> Self {
            Self {
                secs: value.as_secs(),
                nanos: value.subsec_nanos(),
            }
        }
    }
}

#[cfg(feature = "gst")]
mod gstreamer_backend;
#[cfg(feature = "mpv")]
mod mpv_backend;
#[cfg(feature = "rusty")]
mod rusty_backend;

mod discord;
mod mpris;
pub mod playlist;

use anyhow::{Context, Result};
use async_trait::async_trait;
use parking_lot::RwLock;
pub use playlist::{Playlist, Status};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use termusiclib::config::{LastPosition, SeekStep, Settings};
use termusiclib::podcast::db::Database as DBPod;
use termusiclib::sqlite::DataBase;
use termusiclib::track::{MediaType, Track};
use termusiclib::utils::get_app_config_path;
use tokio::runtime::Handle;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

#[macro_use]
extern crate log;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendSelect {
    #[cfg(feature = "mpv")]
    Mpv,
    #[cfg(feature = "rusty")]
    Rusty,
    #[cfg(feature = "gst")]
    GStreamer,
    /// Create a new Backend with default backend ordering
    ///
    /// Order:
    /// - [`RustyBackend`](rusty_backend::RustyBackend) (feature `rusty`)
    /// - [`GstreamerBackend`](gstreamer_backend::GStreamerBackend) (feature `gst`)
    /// - [`MpvBackend`](mpv_backend::MpvBackend) (feature `mpv`)
    /// - Compile Error
    #[default]
    Default,
}

/// Enum to choose backend at runtime
#[non_exhaustive]
pub enum Backend {
    #[cfg(feature = "mpv")]
    Mpv(mpv_backend::MpvBackend),
    #[cfg(feature = "rusty")]
    Rusty(rusty_backend::RustyBackend),
    #[cfg(feature = "gst")]
    GStreamer(gstreamer_backend::GStreamerBackend),
}

pub type PlayerCmdReciever = UnboundedReceiver<PlayerCmd>;
pub type PlayerCmdSender = UnboundedSender<PlayerCmd>;

impl Backend {
    /// Create a new Backend based on `backend`([`BackendSelect`])
    fn new_select(backend: BackendSelect, config: &Settings, cmd_tx: PlayerCmdSender) -> Self {
        match backend {
            #[cfg(feature = "mpv")]
            BackendSelect::Mpv => Self::new_mpv(config, cmd_tx),
            #[cfg(feature = "rusty")]
            BackendSelect::Rusty => Self::new_rusty(config, cmd_tx),
            #[cfg(feature = "gst")]
            BackendSelect::GStreamer => Self::new_gstreamer(config, cmd_tx),
            BackendSelect::Default => Self::new_default(config, cmd_tx),
        }
    }

    /// Create a new Backend with default backend ordering
    ///
    /// For the order see [`BackendSelect::Default`]
    #[allow(unreachable_code)]
    fn new_default(config: &Settings, cmd_tx: PlayerCmdSender) -> Self {
        #[cfg(feature = "rusty")]
        return Self::new_rusty(config, cmd_tx);
        #[cfg(feature = "gst")]
        return Self::new_gstreamer(config, cmd_tx);
        #[cfg(feature = "mpv")]
        return Self::new_mpv(config, cmd_tx);

        #[cfg(not(any(feature = "rusty", feature = "gst", feature = "mpv")))]
        compile_error!("No useable backend feature!");
    }

    /// Explicitly choose Backend [`RustyBackend`](rusty_backend::RustyBackend)
    #[cfg(feature = "rusty")]
    fn new_rusty(config: &Settings, cmd_tx: PlayerCmdSender) -> Self {
        info!("Using Backend \"rusty\"");
        Self::Rusty(rusty_backend::RustyBackend::new(config, cmd_tx))
    }

    /// Explicitly choose Backend [`GstreamerBackend`](gstreamer_backend::GStreamerBackend)
    #[cfg(feature = "gst")]
    fn new_gstreamer(config: &Settings, cmd_tx: PlayerCmdSender) -> Self {
        info!("Using Backend \"GStreamer\"");
        Self::GStreamer(gstreamer_backend::GStreamerBackend::new(config, cmd_tx))
    }

    /// Explicitly choose Backend [`MpvBackend`](mpv_backend::MpvBackend)
    #[cfg(feature = "mpv")]
    fn new_mpv(config: &Settings, cmd_tx: PlayerCmdSender) -> Self {
        info!("Using Backend \"mpv\"");
        Self::Mpv(mpv_backend::MpvBackend::new(config, cmd_tx))
    }

    #[must_use]
    pub fn as_player(&self) -> &dyn PlayerTrait {
        match self {
            #[cfg(feature = "mpv")]
            Backend::Mpv(v) => v,
            #[cfg(feature = "rusty")]
            Backend::Rusty(v) => v,
            #[cfg(feature = "gst")]
            Backend::GStreamer(v) => v,
        }
    }

    #[must_use]
    pub fn as_player_mut(&mut self) -> &mut (dyn PlayerTrait + Send) {
        match self {
            #[cfg(feature = "mpv")]
            Backend::Mpv(v) => v,
            #[cfg(feature = "rusty")]
            Backend::Rusty(v) => v,
            #[cfg(feature = "gst")]
            Backend::GStreamer(v) => v,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum PlayerCmd {
    AboutToFinish,
    CycleLoop,
    Eos,
    GetProgress,
    PlaySelected,
    SkipPrevious,
    Pause,
    Play,
    ProcessID,
    Quit,
    ReloadConfig,
    ReloadPlaylist,
    SeekBackward,
    SeekForward,
    SkipNext,
    SpeedDown,
    SpeedUp,
    Tick,
    ToggleGapless,
    TogglePause,
    VolumeDown,
    VolumeUp,
}

pub type SharedSettings = Arc<RwLock<Settings>>;

#[allow(clippy::module_name_repetitions)]
pub struct GeneralPlayer {
    pub backend: Backend,
    pub playlist: Playlist,
    pub config: SharedSettings,
    pub current_track_updated: bool,
    pub mpris: Option<mpris::Mpris>,
    pub discord: discord::Rpc,
    pub db: DataBase,
    pub db_podcast: DBPod,
    pub cmd_tx: PlayerCmdSender,
}

impl GeneralPlayer {
    /// Create a new [`GeneralPlayer`], with the selected `backend`
    ///
    /// # Errors
    ///
    /// - if connecting to the database fails
    /// - if config path creation fails
    pub fn new_backend(
        backend: BackendSelect,
        config: Settings,
        cmd_tx: PlayerCmdSender,
    ) -> Result<Self> {
        let backend = Backend::new_select(backend, &config, cmd_tx.clone());

        let db_path = get_app_config_path().with_context(|| "failed to get podcast db path.")?;

        let db_podcast =
            DBPod::connect(&db_path).with_context(|| "error connecting to podcast db.")?;
        let db = DataBase::new(&config);

        let config = Arc::new(RwLock::new(config));
        let playlist = Playlist::new(config.clone()).unwrap_or_default();
        let mpris = if config.read().player_use_mpris {
            Some(mpris::Mpris::new(cmd_tx.clone()))
        } else {
            None
        };

        Ok(Self {
            backend,
            playlist,
            config,
            mpris,
            discord: discord::Rpc::default(),
            db,
            db_podcast,
            cmd_tx,
            current_track_updated: false,
        })
    }

    /// Create a new [`GeneralPlayer`], with the [`BackendSelect::Default`] backend
    ///
    /// # Errors
    ///
    /// - if connecting to the database fails
    /// - if config path creation fails
    pub fn new(config: Settings, cmd_tx: PlayerCmdSender) -> Result<Self> {
        Self::new_backend(BackendSelect::Default, config, cmd_tx)
    }

    /// Reload the config from file, on fail continue to use the old
    ///
    /// # Errors
    ///
    /// - if Config could not be parsed
    pub fn reload_config(&mut self) -> Result<()> {
        info!("Reloading config");
        let mut config = self.config.write();
        config.load()?;

        if config.player_use_mpris && self.mpris.is_none() {
            // start mpris if new config has it enabled, but is not active yet
            let mut mpris = mpris::Mpris::new(self.cmd_tx.clone());
            // actually set the metadata of the currently playing track, otherwise the controls will work but no title or coverart will be set until next track
            if let Some(track) = self.playlist.current_track() {
                mpris.add_and_play(track);
            }
            // the same for volume
            mpris.update_volume(self.volume());
            self.mpris.replace(mpris);
        } else if !config.player_use_mpris && self.mpris.is_some() {
            // stop mpris if new config does not have it enabled, but is currently active
            self.mpris.take();
        }

        info!("Config Reloaded");

        Ok(())
    }

    fn get_player(&self) -> &dyn PlayerTrait {
        self.backend.as_player()
    }

    fn get_player_mut(&mut self) -> &mut (dyn PlayerTrait + Send) {
        self.backend.as_player_mut()
    }

    pub fn toggle_gapless(&mut self) -> bool {
        let new_gapless = !self.backend.as_player().gapless();
        self.backend.as_player_mut().set_gapless(new_gapless);
        self.config.write().player_gapless = new_gapless;
        new_gapless
    }

    /// Requires that the function is called on a thread with a entered tokio runtime
    pub fn start_play(&mut self) {
        if self.playlist.is_stopped() | self.playlist.is_paused() {
            self.playlist.set_status(Status::Running);
        }

        self.playlist.proceed();

        if let Some(track) = self.playlist.current_track() {
            let track = track.clone();

            info!("Starting Track {:#?}", track);

            if self.playlist.has_next_track() {
                self.playlist.set_next_track(None);
                self.current_track_updated = true;
                info!("gapless next track played");
                #[cfg(feature = "rusty")]
                #[allow(irrefutable_let_patterns)]
                if let Backend::Rusty(ref mut backend) = self.backend {
                    backend.message_on_end();
                }
                self.add_and_play_mpris_discord();
                return;
            }

            self.current_track_updated = true;
            let wait = async {
                self.add_and_play(&track).await;
            };
            Handle::current().block_on(wait);

            self.add_and_play_mpris_discord();
            self.player_restore_last_position();
            #[cfg(feature = "rusty")]
            #[allow(irrefutable_let_patterns)]
            if let Backend::Rusty(ref mut backend) = self.backend {
                backend.message_on_end();
            }
        }
    }

    fn add_and_play_mpris_discord(&mut self) {
        if let Some(track) = self.playlist.current_track() {
            let config = self.config.read();
            if let Some(ref mut mpris) = self.mpris {
                mpris.add_and_play(track);
            }

            if config.player_use_discord {
                self.discord.update(track);
            }
        }
    }
    pub fn enqueue_next_from_playlist(&mut self) {
        if self.playlist.next_track().is_some() {
            return;
        }

        let track = match self.playlist.fetch_next_track() {
            Some(t) => t.clone(),
            None => return,
        };

        self.playlist.set_next_track(Some(&track));
        self.get_player_mut().enqueue_next(&track);

        info!("Next track enqueued: {:#?}", track);
    }

    pub fn next(&mut self) {
        if self.playlist.current_track().is_some() {
            info!("skip route 1 which is in most cases.");
            self.playlist.set_next_track(None);
            self.get_player_mut().skip_one();
        } else {
            info!("skip route 2 cause no current track.");
            self.stop();
            // if let Err(e) = crate::audio_cmd::<()>(PlayerCmd::StartPlay, false) {
            //     debug!("Error in skip route 2: {e}");
            // }
        }
    }
    pub fn previous(&mut self) {
        self.playlist.previous();
        self.playlist.proceed_false();
        self.next();
    }
    pub fn toggle_pause(&mut self) {
        match self.playlist.status() {
            Status::Running => {
                self.get_player_mut().pause();
                let config = self.config.read();
                if let Some(ref mut mpris) = self.mpris {
                    mpris.pause();
                }
                if config.player_use_discord {
                    self.discord.pause();
                }
                self.playlist.set_status(Status::Paused);
            }
            Status::Stopped => {}
            Status::Paused => {
                self.get_player_mut().resume();
                let config = self.config.read();
                if let Some(ref mut mpris) = self.mpris {
                    mpris.resume();
                }
                if config.player_use_discord {
                    let time_pos = self.get_player().position();
                    self.discord.resume(time_pos);
                }
                self.playlist.set_status(Status::Running);
            }
        }
    }

    pub fn pause(&mut self) {
        match self.playlist.status() {
            Status::Running => {
                self.get_player_mut().pause();
                let config = self.config.read();
                if let Some(ref mut mpris) = self.mpris {
                    mpris.pause();
                }
                if config.player_use_discord {
                    self.discord.pause();
                }
                self.playlist.set_status(Status::Paused);
            }
            Status::Stopped | Status::Paused => {}
        }
    }

    pub fn play(&mut self) {
        match self.playlist.status() {
            Status::Running | Status::Stopped => {}
            Status::Paused => {
                self.get_player_mut().resume();
                let config = self.config.read();
                if let Some(ref mut mpris) = self.mpris {
                    mpris.resume();
                }
                if config.player_use_discord {
                    let time_pos = self.get_player().position();
                    self.discord.resume(time_pos);
                }
                self.playlist.set_status(Status::Running);
            }
        }
    }
    /// # Panics
    ///
    /// if the underlying "seek" returns a error (which current never happens)
    pub fn seek_relative(&mut self, forward: bool) {
        let mut offset = match self.config.read().player_seek_step {
            SeekStep::Short => -5_i64,
            SeekStep::Long => -30,
            SeekStep::Auto => {
                if let Some(track) = self.playlist.current_track() {
                    if track.duration().as_secs() >= 600 {
                        -30
                    } else {
                        -5
                    }
                } else {
                    -5
                }
            }
        };
        if forward {
            offset = -offset;
        }
        self.get_player_mut()
            .seek(offset)
            .expect("Error in player seek.");
    }

    #[allow(clippy::cast_sign_loss)]
    pub fn player_save_last_position(&mut self) {
        match self.config.read().player_remember_last_played_position {
            LastPosition::Yes => {
                if let Some(track) = self.playlist.current_track() {
                    // let time_pos = self.player.position.lock().unwrap();
                    let time_pos = self.get_player().position();
                    match track.media_type {
                        MediaType::Music => self
                            .db
                            .set_last_position(track, time_pos.unwrap_or_default()),
                        MediaType::Podcast => {
                            self.db_podcast
                                .set_last_position(track, time_pos.unwrap_or_default());
                        }
                        MediaType::LiveRadio => {}
                    }
                }
            }
            LastPosition::No => {}
            LastPosition::Auto => {
                if let Some(track) = self.playlist.current_track() {
                    // 10 minutes
                    if track.duration().as_secs() >= 600 {
                        // let time_pos = self.player.position.lock().unwrap();
                        let time_pos = self.get_player().position();
                        match track.media_type {
                            MediaType::Music => self
                                .db
                                .set_last_position(track, time_pos.unwrap_or_default()),
                            MediaType::Podcast => {
                                self.db_podcast
                                    .set_last_position(track, time_pos.unwrap_or_default());
                            }
                            MediaType::LiveRadio => {}
                        }
                    }
                }
            }
        }
    }

    pub fn player_restore_last_position(&mut self) {
        let mut restored = false;
        let last_pos = self.config.read().player_remember_last_played_position;
        match last_pos {
            LastPosition::Yes => {
                if let Some(track) = self.playlist.current_track() {
                    match track.media_type {
                        MediaType::Music => {
                            if let Ok(last_pos) = self.db.get_last_position(track) {
                                self.get_player_mut().seek_to(last_pos);
                                restored = true;
                            }
                        }

                        MediaType::Podcast => {
                            if let Ok(last_pos) = self.db_podcast.get_last_position(track) {
                                self.get_player_mut().seek_to(last_pos);
                                restored = true;
                            }
                        }
                        MediaType::LiveRadio => {}
                    }
                }
            }
            LastPosition::No => {}
            LastPosition::Auto => {
                if let Some(track) = self.playlist.current_track() {
                    // 10 minutes
                    if track.duration().as_secs() >= 600 {
                        match track.media_type {
                            MediaType::Music => {
                                if let Ok(last_pos) = self.db.get_last_position(track) {
                                    self.get_player_mut().seek_to(last_pos);
                                    restored = true;
                                }
                            }

                            MediaType::Podcast => {
                                if let Ok(last_pos) = self.db_podcast.get_last_position(track) {
                                    self.get_player_mut().seek_to(last_pos);
                                    restored = true;
                                }
                            }
                            MediaType::LiveRadio => {}
                        }
                    }
                }
            }
        }

        if restored {
            if let Some(track) = self.playlist.current_track() {
                self.db.set_last_position(track, Duration::from_secs(0));
            }
        }
    }
}

#[async_trait]
impl PlayerTrait for GeneralPlayer {
    async fn add_and_play(&mut self, track: &Track) {
        self.get_player_mut().add_and_play(track).await;
    }
    fn volume(&self) -> Volume {
        self.get_player().volume()
    }
    fn add_volume(&mut self, volume: VolumeSigned) -> Volume {
        self.get_player_mut().add_volume(volume)
    }
    fn set_volume(&mut self, volume: Volume) -> Volume {
        self.get_player_mut().set_volume(volume)
    }
    fn pause(&mut self) {
        self.playlist.set_status(Status::Paused);
        self.get_player_mut().pause();
    }
    fn resume(&mut self) {
        self.playlist.set_status(Status::Running);
        self.get_player_mut().resume();
    }
    fn is_paused(&self) -> bool {
        self.get_player().is_paused()
    }
    fn seek(&mut self, secs: i64) -> Result<()> {
        self.get_player_mut().seek(secs)
    }
    fn seek_to(&mut self, position: Duration) {
        self.get_player_mut().seek_to(position);
    }

    fn set_speed(&mut self, speed: Speed) -> Speed {
        self.get_player_mut().set_speed(speed)
    }

    fn add_speed(&mut self, speed: SpeedSigned) -> Speed {
        self.get_player_mut().add_speed(speed)
    }

    fn speed(&self) -> Speed {
        self.get_player().speed()
    }

    fn stop(&mut self) {
        self.playlist.set_status(Status::Stopped);
        self.playlist.set_next_track(None);
        self.playlist.clear_current_track();
        self.get_player_mut().stop();
    }

    fn get_progress(&self) -> Option<PlayerProgress> {
        self.get_player().get_progress()
    }

    fn gapless(&self) -> bool {
        self.get_player().gapless()
    }

    fn set_gapless(&mut self, to: bool) {
        self.get_player_mut().set_gapless(to);
    }

    fn skip_one(&mut self) {
        self.get_player_mut().skip_one();
    }

    fn position(&self) -> Option<PlayerTimeUnit> {
        self.get_player().position()
    }

    fn enqueue_next(&mut self, track: &Track) {
        self.get_player_mut().enqueue_next(track);
    }

    fn media_info(&self) -> MediaInfo {
        self.get_player().media_info()
    }
}

/// The primitive in which time (current position / total duration) will be stored as
pub type PlayerTimeUnit = Duration;

/// Struct to keep both values with a name, as tuples cannot have named fields
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerProgress {
    pub position: Option<PlayerTimeUnit>,
    /// Total duration of the currently playing track, if there is a known total duration
    pub total_duration: Option<PlayerTimeUnit>,
}

impl From<crate::player::PlayerTime> for PlayerProgress {
    fn from(value: crate::player::PlayerTime) -> Self {
        Self {
            position: value.position.map(std::convert::Into::into),
            total_duration: value.total_duration.map(std::convert::Into::into),
        }
    }
}

impl From<PlayerProgress> for crate::player::PlayerTime {
    fn from(value: PlayerProgress) -> Self {
        Self {
            position: value.position.map(std::convert::Into::into),
            total_duration: value.total_duration.map(std::convert::Into::into),
        }
    }
}

/// Some information that may be available from the backend
/// This is different from [`Track`] as this is everything parsed from the decoder's metadata
/// and [`Track`] stores some different extra stuff
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MediaInfo {
    /// The title of the current media playing (if present)
    pub media_title: Option<String>,
}

pub type Volume = u16;
/// The type of [`Volume::saturating_add_signed`]
pub type VolumeSigned = i16;
pub type Speed = i32;
// yes this is currently the same as speed, but for consistentcy with VolumeSigned (and maybe other types)
pub type SpeedSigned = Speed;

pub const MIN_SPEED: Speed = 1;
pub const MAX_SPEED: Speed = 30;

#[allow(clippy::module_name_repetitions)]
#[async_trait]
pub trait PlayerTrait {
    /// Add the given track, skip to it (if not already) and start playing
    async fn add_and_play(&mut self, track: &Track);
    /// Get the currently set volume
    fn volume(&self) -> Volume;
    /// Add a relative amount to the current volume
    ///
    /// Returns the new volume
    fn add_volume(&mut self, volume: VolumeSigned) -> Volume {
        let volume = self.volume().saturating_add_signed(volume);
        self.set_volume(volume)
    }
    /// Set the volume to a specific amount.
    ///
    /// Returns the new volume
    fn set_volume(&mut self, volume: Volume) -> Volume;
    fn pause(&mut self);
    fn resume(&mut self);
    fn is_paused(&self) -> bool;
    /// Seek relatively to the current time
    ///
    /// # Errors
    ///
    /// Depending on different backend, there could be different errors during seek.
    fn seek(&mut self, secs: i64) -> Result<()>;
    // TODO: sync return types between "seek" and "seek_to"?
    /// Seek to a absolute position
    fn seek_to(&mut self, position: Duration);
    /// Get current track time position
    fn get_progress(&self) -> Option<PlayerProgress>;
    /// Set the speed to a specific amount.
    ///
    /// Returns the new speed
    fn set_speed(&mut self, speed: Speed) -> Speed;
    /// Add a relative amount to the current speed
    ///
    /// Returns the new speed
    fn add_speed(&mut self, speed: SpeedSigned) -> Speed {
        // NOTE: the clamping should likely be done in `set_speed` instead of here
        let speed = (self.speed() + speed).clamp(MIN_SPEED, MAX_SPEED);
        self.set_speed(speed)
    }
    /// Get the currently set speed
    fn speed(&self) -> Speed;
    fn stop(&mut self);
    fn gapless(&self) -> bool;
    fn set_gapless(&mut self, to: bool);
    fn skip_one(&mut self);
    /// Quickly access the position.
    ///
    /// This should ALWAYS match up with [`PlayerTrait::get_progress`]'s `.position`!
    fn position(&self) -> Option<PlayerTimeUnit> {
        self.get_progress()?.position
    }
    /// Add the given URI to be played, but do not skip currently playing track
    fn enqueue_next(&mut self, track: &Track);
    /// Get info of the current media
    fn media_info(&self) -> MediaInfo;
}
