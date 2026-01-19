import {
  Inter,
  Inter_Tight,
  Jersey_25,
  Fira_Code,
  Google_Sans_Code,
} from "next/font/google";

export const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});
export const interTight = Inter_Tight({
  subsets: ["latin"],
  variable: "--font-inter-tight",
});
export const googleSansCode = Google_Sans_Code({
  subsets: ["latin"],
  variable: "--font-google-sans-code",
  weight: "400",
});
export const jersey25 = Jersey_25({
  subsets: ["latin"],
  variable: "--font-jersey-25",
  weight: "400",
});
