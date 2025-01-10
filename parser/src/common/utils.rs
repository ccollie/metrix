use std::fmt;
use std::fmt::{Display, Formatter};
use std::hash::Hasher;

/// Returns the mantissa, exponent and sign as integers.
pub fn integer_decode(val: f64) -> (u64, i16, i8) {
    let bits: u64 = val.to_bits();
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };

    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}

pub fn hash_f64<H: Hasher>(state: &mut H, value: f64) {
    let (mantissa, exponent, sign) = integer_decode(value);
    state.write_u64(mantissa);
    state.write_i16(exponent);
    state.write_i8(sign)
}

pub fn write_number(f: &mut Formatter<'_>, value: f64) -> fmt::Result {
    if value.is_finite() {
        write!(f, "{}", value)
    } else if value.is_nan() {
        write!(f, "NaN")
    } else if value.is_sign_positive() {
        write!(f, "+Inf")
    } else {
        write!(f, "-Inf")
    }
}

pub(crate) fn write_comma_separated<T: Display>(
    values: impl Iterator<Item = T>,
    f: &mut Formatter,
    use_parens: bool,
) -> Result<(), fmt::Error> {
    if use_parens {
        write!(f, "(")?;
    }
    for (i, arg) in values.enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", arg)?;
    }
    if use_parens {
        write!(f, ")")?;
    }
    Ok(())
}
