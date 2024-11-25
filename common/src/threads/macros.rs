use chili::Scope;

macro_rules! join_all {
    ($e1:expr, $e2:expr) => {
        Scope::global().join($e1, $e2)
    };
    ($e1:expr, $e2:expr, $e3:expr) => {
        use chili::Scope;
        let (x, (y, z)) = Scope::global().join($e1, || join_all!($e2, $e3));
        (x, y, z)
    };
    ($e1:expr, $e2:expr, $e3:expr, $e4:expr) => {{
        use chili::Scope;
        let ((a, b), (c, d)) = Scope::global().join(|| join_all!($e1, $e2), || join_all!($e3, $e4));
        (a, b, c, d)
    }};
    ($e1:expr, $e2:expr, $e3:expr, $e4:expr, $e5:expr) => {{
        use chili::Scope;
        let ((a, b, c, d), e) = Scope::global().join(|| join_all!($e1, $e2, $e3, $e4), $e5);
        (a, b, c, d, e)
    }};
    ($e1:expr, $e2:expr, $e3:expr, $e4:expr, $e5:expr, $e6:expr) => {{
        use chili::Scope;
        let ((a, b, c), (d, e, f)) = Scope::global().join(|| join_all!($e1, $e2, $e3), || join_all!($e4, $e5, $e6));
        (a, b, c, d, e, f)
    }};
}