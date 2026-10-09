// SPDX-License-Identifier: GPL-2.0-or-later

//! The sound cards `-audio model=` can add, QEMU's `hw/audio/model.c`.

use std::sync::Mutex;

/// A card `-audio model=` can name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioModel {
    /// The name on the command line.
    pub name: &'static str,
    /// The description `-audio model=help` prints.
    pub descr: &'static str,
    /// The device type to create, such as `AC97`.
    pub typename: &'static str,
}

struct Models {
    list: Vec<AudioModel>,
    selected: Option<(AudioModel, String)>,
}

static MODELS: Mutex<Models> = Mutex::new(Models { list: Vec::new(), selected: None });

fn models() -> std::sync::MutexGuard<'static, Models> {
    MODELS.lock().unwrap_or_else(|e| e.into_inner())
}

/// `audio_register_model()`. QEMU has room for eight.
pub fn register(name: &'static str, descr: &'static str, typename: &'static str) {
    let mut m = models();
    assert!(m.list.len() < 8);
    if m.list.iter().all(|c| c.name != name) {
        m.list.push(AudioModel { name, descr, typename });
    }
}

/// What `audio_print_available_models()` prints.
pub fn available_models_text() -> String {
    let m = models();
    if m.list.is_empty() {
        return "Machine has no user-selectable audio hardware (it may or may not have \
                always-present audio hardware).\n"
            .to_string();
    }
    let mut s = String::from("Valid audio device model names:\n");
    for c in &m.list {
        s.push_str(&format!("{:<11} {}\n", c.name, c.descr));
    }
    s
}

/// Why [`set_model`] refused a model. QEMU prints the message and exits with status 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SetModelError {
    /// A second `-audio model=`.
    Twice,
    /// No card has the name. The text is the list of models to print after the error.
    Unknown(String),
}

/// `audio_set_model()`: the card to create for `audiodev` once the machine exists.
pub fn set_model(name: &str, audiodev: &str) -> Result<(), SetModelError> {
    let mut m = models();
    if m.selected.is_some() {
        return Err(SetModelError::Twice);
    }
    match m.list.iter().find(|c| c.name == name).cloned() {
        Some(c) => {
            m.selected = Some((c, audiodev.to_string()));
            Ok(())
        }
        None => {
            drop(m);
            Err(SetModelError::Unknown(available_models_text()))
        }
    }
}

/// The card `-audio model=` picked and its audiodev, for `audio_model_init()`.
pub fn selected() -> Option<(AudioModel, String)> {
    models().selected.clone()
}
