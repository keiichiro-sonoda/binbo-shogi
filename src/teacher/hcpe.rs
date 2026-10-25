use shogi_core::{Color, Move, Piece, PieceKind, Square};
use std::fmt;
use std::io::{self, Write};
use yasai::Position;

pub const HCP_SIZE: usize = 32;
pub const HCPE_ENTRY_SIZE: usize = 38;

const HAND_PIECES: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

#[repr(i8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HcpeGameResult {
    Draw = 0,
    BlackWin = 1,
    WhiteWin = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HcpeEntry {
    pub hcp: [u8; HCP_SIZE],
    pub eval: i16,
    pub best_move16: u16,
    pub game_result: HcpeGameResult,
}

impl HcpeEntry {
    pub fn from_position(
        pos: &Position,
        eval: i16,
        best_move: Move,
        game_result: HcpeGameResult,
    ) -> Result<Self, HcpeError> {
        Ok(Self {
            hcp: huffman_coded_pos(pos)?,
            eval,
            best_move16: apery_move16(best_move),
            game_result,
        })
    }

    pub fn encode(self) -> [u8; HCPE_ENTRY_SIZE] {
        let mut bytes = [0_u8; HCPE_ENTRY_SIZE];
        bytes[..HCP_SIZE].copy_from_slice(&self.hcp);
        bytes[32..34].copy_from_slice(&self.eval.to_le_bytes());
        bytes[34..36].copy_from_slice(&self.best_move16.to_le_bytes());
        bytes[36] = self.game_result as i8 as u8;
        bytes[37] = 0;
        bytes
    }
}

#[derive(Debug)]
pub enum HcpeError {
    MissingKing(Color),
    InvalidPieceBoxCount { piece: PieceKind, count: i32 },
    InvalidHcpBitLength { bits: usize },
    Io(io::Error),
}

impl fmt::Display for HcpeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingKing(color) => write!(f, "missing king: {color:?}"),
            Self::InvalidPieceBoxCount { piece, count } => {
                write!(f, "invalid piece box count for {piece:?}: {count}")
            }
            Self::InvalidHcpBitLength { bits } => {
                write!(f, "invalid hcp bit length: {bits}, expected 256")
            }
            Self::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for HcpeError {}

impl From<io::Error> for HcpeError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

pub fn write_entries<W: Write>(writer: &mut W, entries: &[HcpeEntry]) -> Result<(), HcpeError> {
    for entry in entries {
        writer.write_all(&entry.encode())?;
    }
    Ok(())
}

pub fn huffman_coded_pos(pos: &Position) -> Result<[u8; HCP_SIZE], HcpeError> {
    let mut writer = BitWriter::new();

    writer.write_bit(matches!(pos.side_to_move(), Color::White));
    writer.write_bits(
        u16::from(square_hcp_index(find_king(pos, Color::Black)?)),
        7,
    );
    writer.write_bits(
        u16::from(square_hcp_index(find_king(pos, Color::White)?)),
        7,
    );

    let mut piece_box_counts = [18_i32, 4, 4, 4, 4, 2, 2];

    for square in Square::all() {
        let piece = pos.piece_at(square);
        if piece
            .map(|piece| piece.piece_kind() == PieceKind::King)
            .unwrap_or(false)
        {
            continue;
        }

        let code = board_code(piece);
        writer.write_bits(code.code, code.bits);

        if let Some(piece) = piece {
            let base = hand_base_piece_kind(piece.piece_kind());
            piece_box_counts[piece_box_index(base)] -= 1;
        }
    }

    for color in [Color::Black, Color::White] {
        let hand = pos.hand(color);
        for piece_kind in HAND_PIECES {
            let count = hand.count(piece_kind).unwrap_or(0);
            let code = hand_code(Piece::new(piece_kind, color));
            for _ in 0..count {
                writer.write_bits(code.code, code.bits);
            }
            piece_box_counts[piece_box_index(piece_kind)] -= i32::from(count);
        }
    }

    for piece_kind in HAND_PIECES {
        let count = piece_box_counts[piece_box_index(piece_kind)];
        if count < 0 {
            return Err(HcpeError::InvalidPieceBoxCount {
                piece: piece_kind,
                count,
            });
        }
        let code = piece_box_code(piece_kind);
        for _ in 0..count {
            writer.write_bits(code.code, code.bits);
        }
    }

    if writer.cursor != 256 {
        return Err(HcpeError::InvalidHcpBitLength {
            bits: writer.cursor,
        });
    }
    Ok(writer.data)
}

pub fn apery_move16(mv: Move) -> u16 {
    const PROMOTE: u16 = 1 << 14;
    const DROP_FROM_BASE: u16 = 80;

    match mv {
        Move::Normal { from, to, promote } => {
            let promote = if promote { PROMOTE } else { 0 };
            u16::from(square_hcp_index(to)) | (u16::from(square_hcp_index(from)) << 7) | promote
        }
        Move::Drop { piece, to } => {
            let piece_kind = apery_drop_piece_kind(piece.piece_kind());
            u16::from(square_hcp_index(to)) | ((u16::from(piece_kind) + DROP_FROM_BASE) << 7)
        }
    }
}

fn apery_drop_piece_kind(piece_kind: PieceKind) -> u8 {
    match piece_kind {
        PieceKind::Pawn => 1,
        PieceKind::Lance => 2,
        PieceKind::Knight => 3,
        PieceKind::Silver => 4,
        PieceKind::Bishop => 5,
        PieceKind::Rook => 6,
        PieceKind::Gold => 7,
        _ => unreachable!("not a droppable piece kind: {piece_kind:?}"),
    }
}

#[derive(Debug, Clone, Copy)]
struct HuffmanCode {
    code: u16,
    bits: u8,
}

struct BitWriter {
    data: [u8; HCP_SIZE],
    cursor: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            data: [0; HCP_SIZE],
            cursor: 0,
        }
    }

    fn write_bit(&mut self, bit: bool) {
        if bit {
            self.data[self.cursor / 8] |= 1 << (self.cursor % 8);
        }
        self.cursor += 1;
    }

    fn write_bits(&mut self, mut value: u16, bits: u8) {
        for _ in 0..bits {
            self.write_bit((value & 1) != 0);
            value >>= 1;
        }
    }
}

fn find_king(pos: &Position, color: Color) -> Result<Square, HcpeError> {
    Square::all()
        .find(|&square| pos.piece_at(square) == Some(Piece::new(PieceKind::King, color)))
        .ok_or(HcpeError::MissingKing(color))
}

fn square_hcp_index(square: Square) -> u8 {
    square.index() - 1
}

fn board_code(piece: Option<Piece>) -> HuffmanCode {
    let Some(piece) = piece else {
        return HuffmanCode {
            code: 0x00,
            bits: 1,
        };
    };
    match (piece.color(), piece.piece_kind()) {
        (Color::Black, PieceKind::Pawn) => HuffmanCode {
            code: 0x01,
            bits: 4,
        },
        (Color::Black, PieceKind::Lance) => HuffmanCode {
            code: 0x03,
            bits: 6,
        },
        (Color::Black, PieceKind::Knight) => HuffmanCode {
            code: 0x07,
            bits: 6,
        },
        (Color::Black, PieceKind::Silver) => HuffmanCode {
            code: 0x0b,
            bits: 6,
        },
        (Color::Black, PieceKind::Bishop) => HuffmanCode {
            code: 0x1f,
            bits: 8,
        },
        (Color::Black, PieceKind::Rook) => HuffmanCode {
            code: 0x3f,
            bits: 8,
        },
        (Color::Black, PieceKind::Gold) => HuffmanCode {
            code: 0x0f,
            bits: 6,
        },
        (Color::Black, PieceKind::ProPawn) => HuffmanCode {
            code: 0x09,
            bits: 4,
        },
        (Color::Black, PieceKind::ProLance) => HuffmanCode {
            code: 0x23,
            bits: 6,
        },
        (Color::Black, PieceKind::ProKnight) => HuffmanCode {
            code: 0x27,
            bits: 6,
        },
        (Color::Black, PieceKind::ProSilver) => HuffmanCode {
            code: 0x2b,
            bits: 6,
        },
        (Color::Black, PieceKind::ProBishop) => HuffmanCode {
            code: 0x9f,
            bits: 8,
        },
        (Color::Black, PieceKind::ProRook) => HuffmanCode {
            code: 0xbf,
            bits: 8,
        },
        (Color::White, PieceKind::Pawn) => HuffmanCode {
            code: 0x05,
            bits: 4,
        },
        (Color::White, PieceKind::Lance) => HuffmanCode {
            code: 0x13,
            bits: 6,
        },
        (Color::White, PieceKind::Knight) => HuffmanCode {
            code: 0x17,
            bits: 6,
        },
        (Color::White, PieceKind::Silver) => HuffmanCode {
            code: 0x1b,
            bits: 6,
        },
        (Color::White, PieceKind::Bishop) => HuffmanCode {
            code: 0x5f,
            bits: 8,
        },
        (Color::White, PieceKind::Rook) => HuffmanCode {
            code: 0x7f,
            bits: 8,
        },
        (Color::White, PieceKind::Gold) => HuffmanCode {
            code: 0x2f,
            bits: 6,
        },
        (Color::White, PieceKind::ProPawn) => HuffmanCode {
            code: 0x0d,
            bits: 4,
        },
        (Color::White, PieceKind::ProLance) => HuffmanCode {
            code: 0x33,
            bits: 6,
        },
        (Color::White, PieceKind::ProKnight) => HuffmanCode {
            code: 0x37,
            bits: 6,
        },
        (Color::White, PieceKind::ProSilver) => HuffmanCode {
            code: 0x3b,
            bits: 6,
        },
        (Color::White, PieceKind::ProBishop) => HuffmanCode {
            code: 0xdf,
            bits: 8,
        },
        (Color::White, PieceKind::ProRook) => HuffmanCode {
            code: 0xff,
            bits: 8,
        },
        (_, PieceKind::King) => panic!("king is encoded as square, not board piece"),
    }
}

fn hand_code(piece: Piece) -> HuffmanCode {
    match piece.as_u8() {
        1 => HuffmanCode {
            code: 0x00,
            bits: 3,
        },
        2 => HuffmanCode {
            code: 0x01,
            bits: 5,
        },
        3 => HuffmanCode {
            code: 0x03,
            bits: 5,
        },
        4 => HuffmanCode {
            code: 0x05,
            bits: 5,
        },
        5 => HuffmanCode {
            code: 0x07,
            bits: 5,
        },
        6 => HuffmanCode {
            code: 0x1f,
            bits: 7,
        },
        7 => HuffmanCode {
            code: 0x3f,
            bits: 7,
        },
        17 => HuffmanCode {
            code: 0x04,
            bits: 3,
        },
        18 => HuffmanCode {
            code: 0x11,
            bits: 5,
        },
        19 => HuffmanCode {
            code: 0x13,
            bits: 5,
        },
        20 => HuffmanCode {
            code: 0x15,
            bits: 5,
        },
        21 => HuffmanCode {
            code: 0x17,
            bits: 5,
        },
        22 => HuffmanCode {
            code: 0x5f,
            bits: 7,
        },
        23 => HuffmanCode {
            code: 0x7f,
            bits: 7,
        },
        other => panic!("piece is not encodable in hcp hand: {other}"),
    }
}

fn piece_box_code(piece_kind: PieceKind) -> HuffmanCode {
    match piece_kind {
        PieceKind::Pawn => HuffmanCode {
            code: 0x02,
            bits: 3,
        },
        PieceKind::Lance => HuffmanCode {
            code: 0x09,
            bits: 5,
        },
        PieceKind::Knight => HuffmanCode {
            code: 0x0b,
            bits: 5,
        },
        PieceKind::Silver => HuffmanCode {
            code: 0x0d,
            bits: 5,
        },
        PieceKind::Gold => HuffmanCode {
            code: 0x1d,
            bits: 5,
        },
        PieceKind::Bishop => HuffmanCode {
            code: 0x0f,
            bits: 7,
        },
        PieceKind::Rook => HuffmanCode {
            code: 0x2f,
            bits: 7,
        },
        _ => panic!("piece kind is not encodable in hcp piece box: {piece_kind:?}"),
    }
}

fn piece_box_index(piece_kind: PieceKind) -> usize {
    match piece_kind {
        PieceKind::Pawn => 0,
        PieceKind::Lance => 1,
        PieceKind::Knight => 2,
        PieceKind::Silver => 3,
        PieceKind::Gold => 4,
        PieceKind::Bishop => 5,
        PieceKind::Rook => 6,
        _ => panic!("piece kind has no piece box slot: {piece_kind:?}"),
    }
}

fn hand_base_piece_kind(piece_kind: PieceKind) -> PieceKind {
    piece_kind.unpromote().unwrap_or(piece_kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{initial_position, parse_position, INITIAL_SFEN};
    use shogi_core::Square;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn hcp_startpos_matches_cshogi_compatible_reference() {
        let pos = initial_position();
        let hcp = huffman_coded_pos(&pos).unwrap();
        assert_eq!(
            hex(&hcp),
            "58a449210cd757217e8e4d212caf427814c2ab109e4d212cd75f213e8e49210c"
        );
    }

    #[test]
    fn hcp_midgame_matches_cshogi_compatible_reference() {
        let pos = parse_position("position startpos moves 7g7f 3c3d").unwrap();
        let hcp = huffman_coded_pos(&pos).unwrap();
        assert_eq!(
            hex(&hcp),
            "58a449210cd757217e8e8d222caf427814c2ab109e4d112cd75f213e8e49210c"
        );
    }

    #[test]
    fn hcp_hand_matches_cshogi_compatible_reference() {
        let pos = parse_position(&format!(
            "position sfen {}",
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPP1/1B5R1/LNSGKGSNL w P 42"
        ))
        .unwrap();
        let hcp = huffman_coded_pos(&pos).unwrap();
        assert_eq!(
            hex(&hcp),
            "59a44981e1fa2ac4cfb12984e555088f427815c2b32984e5fa2bc4c731298401"
        );
    }

    #[test]
    fn move16_matches_apery_cshogi_values() {
        assert_eq!(
            apery_move16(Move::Normal {
                from: Square::new(7, 7).unwrap(),
                to: Square::new(7, 6).unwrap(),
                promote: false,
            }),
            7739
        );
        assert_eq!(
            apery_move16(Move::Normal {
                from: Square::new(2, 7).unwrap(),
                to: Square::new(2, 6).unwrap(),
                promote: false,
            }),
            1934
        );
        assert_eq!(
            apery_move16(Move::Drop {
                piece: Piece::new(PieceKind::Pawn, Color::Black),
                to: Square::new(1, 7).unwrap(),
            }),
            10374
        );

        let to = Square::new(5, 8).unwrap();
        let drop_cases = [
            (PieceKind::Pawn, 10411),
            (PieceKind::Lance, 10539),
            (PieceKind::Knight, 10667),
            (PieceKind::Silver, 10795),
            (PieceKind::Bishop, 10923),
            (PieceKind::Rook, 11051),
            (PieceKind::Gold, 11179),
        ];
        for (piece_kind, expected) in drop_cases {
            assert_eq!(
                apery_move16(Move::Drop {
                    piece: Piece::new(piece_kind, Color::Black),
                    to,
                }),
                expected,
                "{piece_kind:?}"
            );
        }
    }

    #[test]
    fn hcpe_entry_is_38_bytes_and_little_endian() {
        let pos = parse_position(&format!("position sfen {INITIAL_SFEN}")).unwrap();
        let entry = HcpeEntry::from_position(
            &pos,
            -456,
            Move::Normal {
                from: Square::new(2, 7).unwrap(),
                to: Square::new(2, 6).unwrap(),
                promote: false,
            },
            HcpeGameResult::WhiteWin,
        )
        .unwrap();
        let bytes = entry.encode();
        assert_eq!(bytes.len(), HCPE_ENTRY_SIZE);
        assert_eq!(&bytes[32..34], &(-456_i16).to_le_bytes());
        assert_eq!(&bytes[34..36], &1934_u16.to_le_bytes());
        assert_eq!(bytes[36], 2);
        assert_eq!(bytes[37], 0);
    }
}
