/**
 * MIT License
 *
 * tuifeed - Copyright (c) 2021 Christian Visintin
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
use termusiclib::types::{Id, Msg};
use termusicplayback::SharedSettings;
use tuirealm::{
    props::{Alignment, Color},
    Component, Event, MockComponent, NoUserEvent,
};

use crate::ui::model::Model;

use super::{YNConfirm, YNConfirmStyle};

#[derive(MockComponent)]
pub struct QuitPopup {
    component: YNConfirm,
}

impl QuitPopup {
    pub fn new(config: SharedSettings) -> Self {
        let component = YNConfirm::new_with_cb(config, " Are sure you want to quit? ", |config| {
            YNConfirmStyle {
                foreground_color: config
                    .style_color_symbol
                    .library_foreground()
                    .unwrap_or(Color::Yellow),
                background_color: config
                    .style_color_symbol
                    .library_background()
                    .unwrap_or(Color::Reset),
                border_color: config
                    .style_color_symbol
                    .library_border()
                    .unwrap_or(Color::Yellow),
                title_alignment: Alignment::Center,
            }
        });

        Self { component }
    }
}

impl Component<Msg, NoUserEvent> for QuitPopup {
    fn on(&mut self, ev: Event<NoUserEvent>) -> Option<Msg> {
        self.component
            .on(ev, Msg::QuitPopupCloseOk, Msg::QuitPopupCloseCancel)
    }
}

impl Model {
    /// Mount quit popup
    pub fn mount_quit_popup(&mut self) {
        assert!(self
            .app
            .remount(
                Id::QuitPopup,
                Box::new(QuitPopup::new(self.config.clone())),
                vec![]
            )
            .is_ok());
        assert!(self.app.active(&Id::QuitPopup).is_ok());
    }
}
