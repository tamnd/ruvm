// SPDX-License-Identifier: GPL-2.0-or-later

//! The QMP command table and dispatcher, qapi/qmp-registry.c and qapi/qmp-dispatch.c.
//!
//! A [`QmpCommandList`] maps command names to marshalling functions. The generated
//! `register_*` functions in [`crate::commands`] wrap a typed handler into one, so the handler
//! never sees JSON. [`qmp_dispatch`] takes a request as it came off the wire, checks it the way
//! QEMU does, runs the command and builds the response.
//!
//! `C` is whatever the monitor wants to hand each handler, the part `monitor_cur()` plays in
//! QEMU. QEMU's coroutine juggling has no counterpart: a handler is a plain call and the caller
//! decides which thread it runs on.

use std::fmt;
use std::sync::Arc;

use ruvm_base::{Error, ErrorClass, Result};

use crate::visit::{CompatPolicy, compat_policy_input_ok};
use crate::{QDict, QType, QValue};

/// `QmpCommandOptions`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QmpCommandOptions(pub u32);

impl QmpCommandOptions {
    pub const NONE: Self = QmpCommandOptions(0);
    /// `QCO_NO_SUCCESS_RESP`: the command sends no response when it succeeds.
    pub const NO_SUCCESS_RESP: Self = QmpCommandOptions(1 << 0);
    /// `QCO_ALLOW_OOB`: the command may run out of band.
    pub const ALLOW_OOB: Self = QmpCommandOptions(1 << 1);
    /// `QCO_ALLOW_PRECONFIG`: the command works before the machine is ready.
    pub const ALLOW_PRECONFIG: Self = QmpCommandOptions(1 << 2);
    /// `QCO_COROUTINE`: the command may yield.
    pub const COROUTINE: Self = QmpCommandOptions(1 << 3);

    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for QmpCommandOptions {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        QmpCommandOptions(self.0 | rhs.0)
    }
}

/// `QmpCommandFunc`: parses the arguments, runs the handler and turns its result into a value.
/// `None` is a command without `returns`.
pub type QmpCommandFunc<C> =
    Arc<dyn Fn(&C, QDict, &CompatPolicy) -> Result<Option<QValue>> + Send + Sync>;

/// `QmpCommand`.
pub struct QmpCommand<C> {
    pub name: String,
    pub func: QmpCommandFunc<C>,
    pub options: QmpCommandOptions,
    /// The special features of the command, `QAPI_DEPRECATED` and `QAPI_UNSTABLE`.
    pub features: u64,
    pub enabled: bool,
    pub disable_reason: Option<String>,
}

impl<C> fmt::Debug for QmpCommand<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QmpCommand")
            .field("name", &self.name)
            .field("options", &self.options)
            .field("features", &self.features)
            .field("enabled", &self.enabled)
            .finish_non_exhaustive()
    }
}

impl<C> Clone for QmpCommand<C> {
    fn clone(&self) -> Self {
        QmpCommand {
            name: self.name.clone(),
            func: self.func.clone(),
            options: self.options,
            features: self.features,
            enabled: self.enabled,
            disable_reason: self.disable_reason.clone(),
        }
    }
}

impl<C> QmpCommand<C> {
    /// `qmp_has_success_response()`.
    pub fn has_success_response(&self) -> bool {
        !self.options.contains(QmpCommandOptions::NO_SUCCESS_RESP)
    }
}

/// `QmpCommandList`. Commands keep the order they were registered in, which is the order
/// `query-commands` walks them in.
pub struct QmpCommandList<C> {
    cmds: Vec<QmpCommand<C>>,
}

impl<C> fmt::Debug for QmpCommandList<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.cmds.iter().map(|c| &c.name)).finish()
    }
}

impl<C> Clone for QmpCommandList<C> {
    fn clone(&self) -> Self {
        QmpCommandList { cmds: self.cmds.clone() }
    }
}

impl<C> Default for QmpCommandList<C> {
    fn default() -> Self {
        QmpCommandList { cmds: Vec::new() }
    }
}

impl<C> QmpCommandList<C> {
    pub fn new() -> Self {
        Self::default()
    }

    /// `qmp_register_command()`.
    pub fn register(
        &mut self,
        name: &str,
        func: QmpCommandFunc<C>,
        options: QmpCommandOptions,
        features: u64,
    ) {
        assert!(
            !(options.contains(QmpCommandOptions::COROUTINE)
                && options.contains(QmpCommandOptions::ALLOW_OOB)),
            "a command cannot be both a coroutine and out of band"
        );
        self.cmds.push(QmpCommand {
            name: name.to_string(),
            func,
            options,
            features,
            enabled: true,
            disable_reason: None,
        });
    }

    /// `qmp_find_command()`.
    pub fn find(&self, name: &str) -> Option<&QmpCommand<C>> {
        self.cmds.iter().find(|c| c.name == name)
    }

    fn toggle(&mut self, name: &str, enabled: bool, reason: Option<&str>) {
        if let Some(c) = self.cmds.iter_mut().find(|c| c.name == name) {
            c.enabled = enabled;
            c.disable_reason = reason.map(str::to_string);
        }
    }

    /// `qmp_disable_command()`.
    pub fn disable(&mut self, name: &str, reason: Option<&str>) {
        self.toggle(name, false, reason);
    }

    /// `qmp_enable_command()`.
    pub fn enable(&mut self, name: &str) {
        self.toggle(name, true, None);
    }

    /// `qmp_for_each_command()`.
    pub fn iter(&self) -> impl Iterator<Item = &QmpCommand<C>> {
        self.cmds.iter()
    }

    pub fn len(&self) -> usize {
        self.cmds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cmds.is_empty()
    }
}

/// `qmp_error_response()`.
pub fn qmp_error_response(err: &Error) -> QDict {
    let error = QDict::new().with("class", err.class().as_str()).with("desc", err.message());
    QDict::new().with("error", error)
}

/// `qmp_is_oob()`: whether a request asks to run out of band.
pub fn qmp_is_oob(dict: &QDict) -> bool {
    dict.contains_key("exec-oob") && !dict.contains_key("execute")
}

/// `qmp_dispatch_check_obj()`.
fn check_obj(dict: &QDict, allow_oob: bool) -> Result<()> {
    let mut exec_key: Option<&str> = None;
    for (name, value) in dict.iter() {
        if name == "execute" || (name == "exec-oob" && allow_oob) {
            if value.qtype() != QType::QString {
                return Err(Error::generic(format!("QMP input member '{name}' must be a string")));
            }
            if let Some(prev) = exec_key {
                return Err(Error::generic(format!(
                    "QMP input member '{name}' clashes with '{prev}'"
                )));
            }
            exec_key = Some(name);
        } else if name == "arguments" {
            if value.qtype() != QType::QDict {
                return Err(Error::generic("QMP input member 'arguments' must be an object"));
            }
        } else if name != "id" {
            return Err(Error::generic(format!("QMP input member '{name}' is unexpected")));
        }
    }
    if exec_key.is_none() {
        return Err(Error::generic("QMP input lacks member 'execute'"));
    }
    Ok(())
}

/// What the dispatcher needs from the rest of the emulator.
#[derive(Debug, Clone, Copy)]
pub struct DispatchEnv<'a> {
    /// The `-compat` policy.
    pub policy: &'a CompatPolicy,
    /// Whether the client enabled the `oob` capability.
    pub allow_oob: bool,
    /// Whether the machine is ready. Before that only `allow-preconfig` commands run, which is
    /// the check `qmp_command_available()` makes in system/qdev-monitor.c. Tools without a
    /// machine pass `true`.
    pub machine_ready: bool,
}

fn run<C>(
    cmds: &QmpCommandList<C>,
    dict: &QDict,
    env: &DispatchEnv<'_>,
    ctx: &C,
) -> Result<Option<QValue>> {
    check_obj(dict, env.allow_oob)?;
    let (command, oob) = match dict.get_str("execute") {
        Some(c) => (c, false),
        None => (dict.get_str("exec-oob").expect("checked above"), true),
    };
    let Some(cmd) = cmds.find(command) else {
        return Err(Error::new(
            ErrorClass::CommandNotFound,
            format!("The command {command} has not been found"),
        ));
    };
    compat_policy_input_ok(
        cmd.features,
        env.policy,
        ErrorClass::CommandNotFound,
        "command",
        command,
    )?;
    if !cmd.enabled {
        let reason = cmd.disable_reason.as_deref().map(|r| format!(": {r}")).unwrap_or_default();
        return Err(Error::new(
            ErrorClass::CommandNotFound,
            format!("Command {command} has been disabled{reason}"),
        ));
    }
    if oob && !cmd.options.contains(QmpCommandOptions::ALLOW_OOB) {
        return Err(Error::generic(format!("The command {command} does not support OOB")));
    }
    if !env.machine_ready && !cmd.options.contains(QmpCommandOptions::ALLOW_PRECONFIG) {
        return Err(Error::generic(format!(
            "The command '{command}' is permitted only after machine initialization has completed"
        )));
    }
    let args = match dict.get("arguments") {
        Some(QValue::Dict(d)) => d.clone(),
        _ => QDict::new(),
    };
    let ret = (cmd.func)(ctx, args, env.policy)?;
    if !cmd.has_success_response() {
        assert!(ret.is_none(), "a command without a success response returned a value");
        return Ok(None);
    }
    Ok(Some(ret.unwrap_or_else(|| QValue::Dict(QDict::new()))))
}

/// `qmp_dispatch()`: runs one request and returns the response, or `None` for a command that
/// sends no response on success.
pub fn qmp_dispatch<C>(
    cmds: &QmpCommandList<C>,
    request: &QValue,
    env: &DispatchEnv<'_>,
    ctx: &C,
) -> Option<QDict> {
    let (result, id) = match request {
        QValue::Dict(dict) => (run(cmds, dict, env, ctx), dict.get("id").cloned()),
        _ => (Err(Error::generic("QMP input must be a JSON object")), None),
    };
    let mut rsp = match result {
        Ok(None) => return None,
        Ok(Some(ret)) => QDict::new().with("return", ret),
        Err(e) => qmp_error_response(&e),
    };
    if let Some(id) = id {
        rsp.put("id", id);
    }
    Some(rsp)
}

/// `qmp_event_build_dict()`: the frame of an event, with the current wall clock time.
pub fn qmp_event_build_dict(name: &str) -> QDict {
    let now =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let ts = QDict::new()
        .with("seconds", now.as_secs() as i64)
        .with("microseconds", i64::from(now.subsec_micros()));
    QDict::new().with("event", name).with("timestamp", ts)
}
