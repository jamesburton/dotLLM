using Mini.Lib;

namespace Mini.App;

public static class Program
{
    public static string Run() => Greeter.Greet(Stamp.Value);
}