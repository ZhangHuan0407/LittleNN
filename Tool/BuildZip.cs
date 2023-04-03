using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Text.RegularExpressions;
using System.Collections.Generic;

namespace BuildZip
{
    internal static class Program
    {
        public static string ZipName = "LittleNN-{Version}.zip";
        public static readonly string ProjectDirectory = new DirectoryInfo(Environment.CurrentDirectory).Parent.FullName.Replace("\\", " / ");
        public static readonly string CSProjectFilePath = ProjectDirectory + "/LittleNN/LittleNN.csproj";
        public static readonly string OutputPath = ProjectDirectory + "/Output";

        public static Dictionary<string, string> ArchiveMap = new Dictionary<string, string>()
        {
            { $"{ProjectDirectory}/LICENSE", $"{OutputPath}/LICENSE" },
            { $"{ProjectDirectory}/LittleNN/bin/Release/net5.0/LittleNN.dll", $"{OutputPath}/LittleNN.dll" },
            { $"{ProjectDirectory}/LittleNN/bin/Release/net5.0/LittleNN.xml", $"{OutputPath}/LittleNN.xml" },
            { $"{ProjectDirectory}/LittleNN/bin/Release/net5.0/LittleNN.pdb", $"{OutputPath}/LittleNN.pdb" },
            // third part
            { $"{ProjectDirectory}/LittleNN/trentsartain/Neural-Network/LICENSE", $"{OutputPath}/trentsartain/Neural-Network/LICENSE" },
        };


        [STAThread]
        static void Main(string[] args)
        {
            Console.WriteLine("BuildZip...");
            string projectContent = File.ReadAllText(CSProjectFilePath);
            if (Regex.Match(projectContent, "<Version>(?<Version>[0-9\\.]+)</Version>") is Match match && match.Success)
            {
                Version version = new Version(match.Groups["Version"].Value);
                ZipName = ZipName.Replace("{Version}", $"{version.Major}.{version.Minor}.{version.Build}");
            }
            else
            {
                throw new Exception("miss version");
            }
		
            
            if (Directory.Exists(OutputPath))
                Directory.Delete(OutputPath, true);
            Directory.CreateDirectory(OutputPath);
            foreach (var pair in ArchiveMap)
            {
                if (!pair.Value.StartsWith(ProjectDirectory))
                    throw new Exception(pair.Value);
                Directory.CreateDirectory(Path.GetDirectoryName(pair.Value));
                byte[] buffer = File.ReadAllBytes(pair.Key);
                File.WriteAllBytes(pair.Value, buffer);
            }

            ZipFile.CreateFromDirectory(OutputPath, Path.Combine(ProjectDirectory, ZipName));
            Console.WriteLine("BuildZip finish");
        }
    }
}
