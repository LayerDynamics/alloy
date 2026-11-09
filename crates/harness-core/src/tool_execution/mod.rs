//! W4 receipt-owned dispatch at the existing W3 harness boundary.
//!
//! This module does not admit, approve, claim or start an invocation. It consumes
//! an existing durable receipt. Registering retains the SAME implementation Arc
//! in W1 and this private dispatch index; no executable handle is exposed.
//! No SessionStore borrow is retained across run. Callers must also release any
//! enclosing session/store/native-slot locks. W6 configuration publication fences
//! and W5 hooks remain separate obligations, not implemented here.

mod executor;

pub use executor::*;
