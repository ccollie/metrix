use chili::Scope;

pub fn par_join_slice<T: Send + Sync, F, R>(slice: &[T], f: F) -> Vec<R>
where F: Fn(&T) -> R + Send + Sync, R: Clone + Send
{
    #[inline]
    fn handle_two<T: Send + Sync, F, R: Send>(s: &mut Scope<'_>, f: &F, first: &T, second: &T) -> (R, R)
    where F: Fn(&T) -> R + Send + Sync
    {
        s.join(|_| f(first), |_| f(second))
    }

    #[inline]
    fn handle_three<T: Send + Sync, F, R: Send>(s: &mut Scope<'_>, f: &F, first: &T, second: &T, third: &T) -> (R, R, R)
    where F: Fn(&T) -> R + Send + Sync
    {
        let ((one, two), three) =
            s.join(|s1| handle_two(s1, f, first, second), |_| f(third));
        (one, two, three)
    }

    #[inline]
    fn handle_four<T: Send + Sync, F, R: Send>(s: &mut Scope<'_>, f: &F, first: &T, second: &T, third: &T, fourth: &T) -> (R, R, R, R)
    where F: Fn(&T) -> R + Send + Sync
    {
        let ((one, two), (three, four)) =
            s.join(|s1| handle_two(s1, f, first, second),
                   |s2| handle_two(s2, f, third, fourth));
        (one, two, three, four)
    }

    #[inline]
    fn handle_five<T: Send + Sync, F, R: Send>(s: &mut Scope<'_>, f: &F, first: &T, second: &T, third: &T, fourth: &T, fifth: &T) -> (R, R, R, R, R)
    where F: Fn(&T) -> R + Send + Sync
    {
        let ((one, two, three), (four, five)) =
            s.join(|s1| handle_three(s1, f, first, second, third), |s2| handle_two(s2, f, fourth, fifth));
        (one, two, three, four, five)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn handle_six<T: Send + Sync, F, R: Send>(s: &mut Scope<'_>, f: &F, first: &T, second: &T, third: &T, fourth: &T, fifth: &T, sixth: &T) -> (R, R, R, R, R, R)
    where F: Fn(&T) -> R + Send + Sync
    {
        let ((one, two, three), (four, five, six)) =
            s.join(|s| handle_three(s, f, first, second, third),
                   |s| handle_three(s, f, fourth, fifth, sixth));
        (one, two, three, four, five, six)
    }

    match slice {
        [] => vec![],
        [first] => vec![f(first)],
        [first, second] => {
            let mut scope = Scope::global();
            let (one, two) = handle_two(&mut scope, &f, first, second);
            vec![one, two]
        },
        [first, second, third] => {
            let mut scope = Scope::global();
            let (one, two, three) = handle_three(&mut scope, &f, first, second, third);
            vec![one, two, three]
        },
        [first, second, third, fourth] => {
            let mut scope = Scope::global();
            let (one, two, three, four) = handle_four(&mut scope, &f, first, second, third, fourth);
            vec![one, two, three, four]
        },
        [first, second, third, fourth, fifth] => {
            let mut scope = Scope::global();
            let (one, two, three, four, five) = handle_five(&mut scope, &f, first, second, third, fourth, fifth);
            vec![one, two, three, four, five]
        },
        [first, second, third, fourth, fifth, sixth] => {
            let mut scope = Scope::global();
            let (one, two, three, four, five, six) = handle_six(&mut scope, &f, first, second, third, fourth, fifth, sixth);
            vec![one, two, three, four, five, six]
        },
        [first, second, third, fourth, fifth, sixth, rest @ ..] => {
            let mut scope = Scope::global();
            let (one, two, three, four, five, six) = handle_six(&mut scope, &f, first, second, third, fourth, fifth, sixth);
            let mut result: Vec<R> = Vec::with_capacity(slice.len());
            result.extend_from_slice(&[one, two, three, four, five, six]);

            let mut right = par_join_slice(rest, f);
            result.append(&mut right);
            result
        }
    }
}
