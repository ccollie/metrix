use integer_encoding::{FixedInt, VarInt};

use crate::{EncodingError, EncodingResult};

// the maximum lengths of a varint-encoded N-bit integer.
pub const MAX_VAR_INT_LEN_16: usize = 3;
pub const MAX_VARINT_LEN_32: usize = 5;
pub const MAX_VARINT_LEN_64: usize = 10;


/// marshal_fixed_int appends marshaled v to dst and returns the result.
pub fn marshal_fixed_int<T: FixedInt>(dst: &mut Vec<u8>, v: T) {
    let mut buf = [0_u8; 16];
    v.encode_fixed(&mut buf[..T::ENCODED_SIZE]);
    dst.extend_from_slice(&buf[0..T::ENCODED_SIZE]);
}

/// unmarshal_fixed_int returns unmarshalled int64 from src.
pub fn unmarshal_fixed_int<T: FixedInt>(src: &[u8]) -> EncodingResult<(T, &[u8])> {
    match T::decode_fixed(src) {
        None => Err(EncodingError::new(format!(
            "At least {} bytes required decoding int. Got {}",
            T::ENCODED_SIZE,
            src.len()
        ))),
        Some(v) => Ok((v, &src[T::ENCODED_SIZE..])),
    }
}

/// marshals a usize to dst.
pub fn marshal_usize(dst: &mut Vec<u8>, v: usize) {
    marshal_fixed_int::<usize>(dst, v)
}

/// unmarshal_usize returns unmarshalled usize from src.
pub fn unmarshal_usize(src: &[u8]) -> EncodingResult<(usize, &[u8])> {
    unmarshal_fixed_int::<usize>(src)
}

/// appends marshaled v to dst and returns the result.
pub fn marshal_var_int<T: VarInt>(dst: &mut Vec<u8>, v: T) {
    let len = dst.len();
    dst.resize(len + v.required_space(), 0);
    let _ = v.encode_var(&mut dst[len..]);
}

/// unmarshal_var returns unmarshalled int from src.
pub fn unmarshal_var_int<T: VarInt>(src: &[u8]) -> EncodingResult<(T, &[u8])> {
    match T::decode_var(src) {
        Some((v, ofs)) => Ok((v, &src[ofs..])),
        _ => Err(EncodingError::new("Error decoding var int".to_string())),
    }
}

/// appends marshaled v to dst and returns the result.
#[inline]
pub fn marshal_var_i64(dst: &mut Vec<u8>, v: i64) {
    marshal_var_int(dst, v)
}

/// unmarshal_var_i64 returns unmarshalled int64 from src and returns
/// the remaining tail from src.
pub fn unmarshal_var_i64(src: &[u8]) -> EncodingResult<(i64, &[u8])> {
    unmarshal_var_int::<i64>(src)
}

/// appends marshaled v to dst and returns the result.
#[inline]
pub fn marshal_var_u64(dst: &mut Vec<u8>, v: u64) {
    marshal_var_int(dst, v)
}

/// returns unmarshalled u64 from src and returns the remaining tail from src.
pub fn unmarshal_var_u64(src: &[u8]) -> EncodingResult<(u64, &[u8])> {
    unmarshal_var_int::<u64>(src)
}

/// appends marshaled v to dst and returns the result.
#[inline]
pub fn marshal_var_usize(dst: &mut Vec<u8>, v: usize) {
    marshal_var_int(dst, v)
}

/// returns unmarshalled u16 from src and returns the remaining tail from src.
pub fn unmarshal_var_usize(src: &[u8]) -> EncodingResult<(usize, &[u8])> {
    unmarshal_var_int::<usize>(src)
}

/// marshal_bytes appends marshaled b to dst and returns the result.
pub fn marshal_bytes(b: &[u8], dst: &mut Vec<u8>) {
    let len = b.len() as u64;
    dst.reserve(b.len() + 10);
    len.encode_var(dst);
    if len > 0 {
        dst.extend_from_slice(b);
    }
}

/// unmarshal_bytes returns unmarshalled bytes from src.
/// returns (bytes, remaining tail from src).
pub fn unmarshal_bytes(src: &[u8]) -> EncodingResult<(&[u8], &[u8])> {
    let (len, n) = u64::decode_var(&src[0..]).ok_or_else(|| EncodingError {
        description: "unable to decode".into(),
    })?;
    let byte_len = src.len();
    let tail = &src[n..];

    if tail.len() < byte_len {
        let description = format!(
            "src is too short for reading string with size {n}; src.len()={}",
            src.len()
        );
        return Err(EncodingError { description });
    }
    let len = len as usize;
    let str_bytes = &tail[..len];
    let tail = &tail[len..];
    Ok((str_bytes, tail))
}
