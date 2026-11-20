import "./globals.css";

export const metadata = {
  title: "SignCaption",
  description: "Live captions from ASL signs, right in your browser.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
