using Mini.Lib;

namespace Mini.App;

public static class Program
{
    public static string Run() => Greeter.Greet(Stamp.Value);

    // Overload-resolution failure (int for string): an invocation that does not bind, though it has a candidate.
    public static string Wrong() => Greeter.Greet(1);
}