namespace Mini.Lib;

public static class Greeter
{
#if MINI_GREETING
    public static string Greet(string name) => "Hello, " + name;
#endif

    public static string Name => "Greeter";

    public static int Major => Mini.BuildInfo.Major + LibInfo.Minor;
}